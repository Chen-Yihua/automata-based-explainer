# Local Explanation for Black-Box Sequential Models

本專案為碩士論文《Local Explanation for Black-Box Sequential Models》的程式碼實作。主要在學習一個精簡的 **決定性有限自動機(DFA)** 來解釋黑箱模型的局部決策邏輯。

**方法概述**：首先透過被動學習演算法（RPNI），從目標序列附近的擾動樣本建構初始 DFA；接著以 **Beam Search** 搭配 **KL-LUCB** 反覆執行 Delete、Merge、Delta 三種操作來生成候選 DFA。KL-LUCB 負責在有限取樣預算下，判斷哪些候選解值得保留。最終從搜尋歷史中選擇滿足使用者指定的一致率門檻下，狀態數最少的自動機。

**實驗設計涵蓋**：
- 三項 regular-language 任務（Secure Handshake、Document Release Workflow、Navigation Workflow），以 ground-truth DFA 作為黑盒 teacher。
- 三項真實世界任務（MNIST 筆畫序列、ECG5000、Wafer 感測序列），以訓練完成的 RNN 分類器作為黑盒 teacher。
- 在相同的精簡操作與評估候選次數下，與三種啟發式搜尋方法進行比較：**Simulated Annealing (SA)**、**Genetic Algorithm (GA)**、**Particle Swarm Optimization (PSO)**。

**完整的實驗重現步驟，請參見 [`RUNNING.md`](RUNNING.md)。**

---

## 專案結構

```text
anchor-llm/
├── src/
│   ├── explainer/automata_beam.py   # Beam Search + KL-LUCB（本論文方法）
│   ├── learner/dfa_learner.py       # RPNI 初始化、Delete/Merge/Delta 操作
│   ├── baselines/                   # SA / GA / PSO baseline 與超參數調整
│   └── automaton/                   # DFA 資料結構、一致率指標、讀寫
├── examples/RPNI/
│   ├── run_regular_experiment.py    # Regular-language 任務實驗
│   ├── run_realworld_experiment.py  # 真實世界任務實驗
│   └── run_kllucb_comparison.py     # Beam Search 有/無 KL-LUCB 比較
├── analysis/
│   ├── parse_results.py             # test_result/ 的 log -> summary_table.csv
│   └── make_plots.py                # summary_table.csv -> LaTeX/pgfplots 圖
├── automata/                        # regular-language 任務用的 ground-truth DFA (.dot)
├── models/                          # 真實世界任務用的已訓練 RNN 分類器與資料切分
├── external_modules/Explaining-FA/  # Delta 操作用的 contrastive explanation (CXp) solver
├── test_result/                     # 實驗輸出（執行實驗後產生）
└── requirements.txt
```

## 安裝

```bash
git clone https://github.com/Chen-Yihua/anchor-automata-explainer.git
cd anchor-automata-explainer

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 固定使用 `tensorflow==2.20.0`。若偵測到相容的 NVIDIA 驅動程式，pip 會自動安裝支援 GPU 的版本；若無，則自動退回 CPU 版本，兩種情況皆不需額外安裝 CUDA 工具組或使用 conda-forge。

## 新增一個 language / dataset

**新增 regular automaton**：（需要 `automata/` 下有對應的 `.dot` 檔）

把 `.dot` 檔放進 `automata/`，然後在 `run_regular_experiment.py` 的 `get_languages_config()` 加一筆：

```python
"NewAutomaton": dict(
    automata_name   = "NewAutomaton",
    filename        = "new_automaton.dot",
    agreement_threshold = 0.9,
    delta           = 0.01,
    tau             = 0.05,
    batch_size      = 1000,
    beam_size       = 1,
    init_num_samples= 1000,
    edit_distance   = 5,
    max_evaluations = 3000,
    num_test_instances = 10,   # 或 1
    test_instance   = None,    # 或指定一組 symbol list
    test_instances  = None,
),
```

**新增 real-world dataset**：（需要 `models/` 下有 `<name>_classifier_trained.pth` 和 `<name>_train_test_split.pkl`）

把模型檔放進 `models/`（`<name>_classifier_trained.pth`、`<name>_train_test_split.pkl`），然後在 `run_realworld_experiment.py` 的 `get_languages_config()` 加一筆：

```python
"new_dataset": dict(
    alphabet        = ['A', 'B', 'C'],
    agreement_threshold = 0.85,
    delta           = 0.01,
    tau             = 0.05,
    batch_size      = 1000,
    beam_size       = 1,
    init_num_samples= 500,
    edit_distance   = 3,
    max_length      = 20,
    max_evaluations = 2500,
    num_test_instances = 10,
    test_instance   = None,
    test_instances  = None,
    embedding_dim   = 64,
    hidden_dim      = 256,
    num_layers      = 2,
    dropout         = 0.3,
),
```

跑法都一樣：`--languages <你取的名字>`。

### 測試序列：指定 vs 未指定

想固定測哪些序列，就寫 `test_instance`（一條）或 `test_instances`（多條），程式會直接照跑，`num_test_instances` 會被忽略。

不想自己準備，就把兩個都留 `None`，程式會自動生成 `num_test_instances` 條：
- Regular：從 teacher DFA 隨機走出來的序列。
- Real-world：從 training set 取前面幾條。

## Baseline (SA/GA/PSO) 超參數調整

`src/baselines/tune_baseline_params.py` 對應論文 5.3 節「多組 baseline 參數設定比較」。它在同一份 `shared_init.pkl`（跑實驗時會自動產生）上，對 SA candidate pool size、GA population size、PSO particle 數/candidate pool size 各掃一個小網格，其他設定（操作集合、agreement threshold、evaluation budget）固定不變。

```bash
python src/baselines/tune_baseline_params.py \
    --tune_root regular_0.8_1000 \
    --datasets SecureHandshake,DocumentReleaseWorkflow
```

輸出在 `test_result/tune_fairflow_baselines_<timestamp>/`：`tune_results.csv`、`best_by_algo.csv`（每個 dataset × 演算法的最佳參數）、`summary_top20.txt`。

---

重現論文實驗的操作步驟，請參見 [`RUNNING.md`](RUNNING.md)。
