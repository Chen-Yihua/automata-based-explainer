# models/sequence_classifier.py

# Unified sequence classifier supporting both RNN and CNN models using PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os

class SequenceClassifier:
    """Unified sequence classifier supporting RNN and CNN architectures.
    
    Args:
        model_type (str): 'rnn' for LSTM-based or 'cnn' for CNN-based classifier
        max_len (int): Maximum sequence length
        embedding_dim (int): Embedding dimension
        rnn_units (int): Hidden dimension for RNN (only used if model_type='rnn')
        num_layers (int): Number of RNN layers (only used if model_type='rnn')
        num_filters (int): Number of filters for CNN (only used if model_type='cnn')
        filter_sizes (list): Kernel sizes for CNN (only used if model_type='cnn')
        dropout (float): Dropout rate
        weight_decay (float): L2 regularization weight
        use_attention (bool): Use attention mechanism (only for model_type='rnn')
        device (str): Device to run on ('cuda' or 'cpu')
    """
    def __init__(self, model_type='rnn', max_len=50, embedding_dim=16, rnn_units=32, num_layers=2, 
                 num_filters=64, filter_sizes=None, dropout=0.3, weight_decay=1e-4, use_attention=False, device=None):
        if filter_sizes is None:
            filter_sizes = [3, 4, 5]
        
        self.model_type = model_type
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # RNN-specific parameters
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # CNN-specific parameters
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        
        self.model = None
        self.symbol2id = None
        self.id2symbol = None
        self.num_classes = None

    def fit(self, X, y, epochs=10, batch_size=32):
        X_pad, vocab_size = self._prepare_X(X, fit_vocab=True)
        self.vocab_size = vocab_size
        y = np.array(y)
        num_classes = len(np.unique(y))
        self.num_classes = num_classes
        
        if self.model_type == 'rnn':
            self.model = _TorchRNNClassifier(
                self.vocab_size+1,
                self.embedding_dim,
                self.rnn_units,
                self.num_classes,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_attention=self.use_attention
            ).to(self.device)
        elif self.model_type == 'cnn':
            self.model = _TorchCNNClassifier(
                self.vocab_size+1,
                self.embedding_dim,
                self.num_filters,
                self.filter_sizes,
                self.num_classes,
                dropout=self.dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        X_tensor = torch.LongTensor(X_pad)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=self.weight_decay)
        steps_per_epoch = len(loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.003, epochs=epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.3,
            anneal_strategy='cos'
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for i, (xb, yb) in enumerate(loader):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * xb.size(0)
                total_correct += (torch.argmax(out, dim=1) == yb).sum().item()
                total_samples += xb.size(0)
            avg_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {lr:.6f}")

    def save(self, model_path):
        """Save model and vocabulary to file."""
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        checkpoint = {
            'model_type': self.model_type,
            'model_state': self.model.state_dict(),
            'symbol2id': self.symbol2id,
            'id2symbol': self.id2symbol,
            'num_classes': self.num_classes,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'dropout': self.dropout,
            'weight_decay': self.weight_decay,
        }
        
        if self.model_type == 'rnn':
            checkpoint.update({
                'rnn_units': self.rnn_units,
                'num_layers': self.num_layers,
                'use_attention': self.use_attention
            })
        elif self.model_type == 'cnn':
            checkpoint.update({
                'num_filters': self.num_filters,
                'filter_sizes': self.filter_sizes
            })
        
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path):
        """Load model and vocabulary from file."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_type = checkpoint.get('model_type', 'rnn')  # Default to 'rnn' for backward compatibility
        self.symbol2id = checkpoint['symbol2id']
        self.id2symbol = checkpoint['id2symbol']
        self.num_classes = checkpoint['num_classes']
        self.vocab_size = checkpoint['vocab_size']
        self.max_len = checkpoint['max_len']
        self.embedding_dim = checkpoint['embedding_dim']
        self.dropout = checkpoint['dropout']
        self.weight_decay = checkpoint.get('weight_decay', 1e-4)
        
        if self.model_type == 'rnn':
            self.rnn_units = checkpoint['rnn_units']
            self.num_layers = checkpoint['num_layers']
            self.use_attention = checkpoint.get('use_attention', False)
            self.model = _TorchRNNClassifier(
                self.vocab_size + 1,
                self.embedding_dim,
                self.rnn_units,
                self.num_classes,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_attention=self.use_attention
            ).to(self.device)
        elif self.model_type == 'cnn':
            self.num_filters = checkpoint['num_filters']
            self.filter_sizes = checkpoint['filter_sizes']
            self.model = _TorchCNNClassifier(
                self.vocab_size + 1,
                self.embedding_dim,
                self.num_filters,
                self.filter_sizes,
                self.num_classes,
                dropout=self.dropout
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"Model loaded from {model_path} (model_type={self.model_type})")

    def predict(self, X, batch_size=128):
        X_pad, _ = self._prepare_X(X, fit_vocab=False)
        X_tensor = torch.LongTensor(X_pad).to(self.device)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                logits = self.model(batch)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(batch_preds)
        return np.concatenate(preds)

    def _prepare_X(self, X, fit_vocab=False):
        # Flatten nested sequences and collect all symbols
        seqs = []
        all_symbols = set()
        for seq in X:
            if isinstance(seq, (list, np.ndarray)) and len(seq) > 0:
                if isinstance(seq[0], (list, np.ndarray)):
                    seq = seq[0]
            for s in seq:
                all_symbols.add(s)
            seqs.append(seq)
        # Build vocab if needed
        if fit_vocab or self.symbol2id is None:
            symbol2id = {s: i+1 for i, s in enumerate(sorted(all_symbols, key=str))}  # 0 reserved for padding
            id2symbol = {i: s for s, i in symbol2id.items()}
            self.symbol2id = symbol2id
            self.id2symbol = id2symbol
        # Encode
        seqs_id = [[self.symbol2id.get(s, 0) for s in seq] for seq in seqs]
        # Pad
        X_pad = np.zeros((len(seqs_id), self.max_len), dtype=np.int64)
        n_truncated = 0
        for i, seq in enumerate(seqs_id):
            l = min(len(seq), self.max_len)
            if len(seq) > self.max_len:
                n_truncated += 1
            X_pad[i, :l] = seq[:l]
        if n_truncated:
            # Silent truncation is dangerous here: the label this classifier
            # returns for a truncated sequence describes only its first
            # max_len symbols, but callers (e.g. perturbation sampling) keep
            # pairing that label with the full, untruncated sequence -- a
            # mismatch that corrupts training data without any error. This
            # doesn't change what gets predicted (still the same truncation
            # as before), just makes it visible instead of silent.
            print(
                f"  [WARNING] SequenceClassifier: {n_truncated}/{len(seqs_id)} "
                f"sequence(s) longer than max_len={self.max_len} were truncated "
                f"before prediction -- their labels describe only the first "
                f"{self.max_len} symbols."
            )
        vocab_size = max(self.symbol2id.values()) if self.symbol2id else 1
        return X_pad, vocab_size


class _TorchRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, num_layers=2, dropout=0.3, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            rnn_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        if use_attention:
            # Multi-head self-attention
            self.attention = nn.MultiheadAttention(
                embed_dim=rnn_units * 2,
                num_heads=4,
                batch_first=True,
                dropout=dropout
            )
            self.bn = nn.BatchNorm1d(rnn_units * 2)
            self.dense = nn.Linear(rnn_units * 2, rnn_units * 2)
        else:
            self.bn = nn.BatchNorm1d(rnn_units * 2)
            self.dense = nn.Linear(rnn_units * 2, rnn_units * 2)
        
        self.fc = nn.Linear(rnn_units * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (h_n, _) = self.lstm(x)
        # lstm_out: (batch, seq_len, rnn_units*2)
        # h_n: (num_layers * 2, batch, rnn_units)
        
        if self.use_attention:
            # 使用 attention 聚合所有时间步
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # 取平均池化
            h = attn_out.mean(dim=1)  # (batch, rnn_units*2)
        else:
            # 只用最后的 hidden state
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)  # (batch, rnn_units*2)
        
        h = self.bn(h)
        h = torch.relu(self.dense(h))
        out = self.fc(h)
        return out


class _TorchCNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多個卷積層，不同的 kernel size
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])
        
        self.dropout_layer = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(num_filters * len(filter_sizes))
        self.dense = nn.Linear(num_filters * len(filter_sizes), num_filters * len(filter_sizes))
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len) for Conv1d
        
        # 應用多個卷積層並做 max pooling
        conv_outs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch, num_filters, seq_len)
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch, num_filters, 1)
            pooled = pooled.squeeze(2)  # (batch, num_filters)
            conv_outs.append(pooled)
        
        # 拼接所有卷積結果
        h = torch.cat(conv_outs, dim=1)  # (batch, num_filters * len(filter_sizes))
        h = self.dropout_layer(h)
        h = self.bn(h)
        h = torch.relu(self.dense(h))
        h = self.dropout_layer(h)
        out = self.fc(h)
        return out

# Backward compatibility aliases
SimpleSequenceClassifier = SequenceClassifier
CNNSequenceClassifier = SequenceClassifier