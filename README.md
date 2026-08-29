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

完整的實驗設定、執行方式、SA/GA/PSO 超參數調整流程，請參見 [`RUNNING.md`](RUNNING.md)。

```
