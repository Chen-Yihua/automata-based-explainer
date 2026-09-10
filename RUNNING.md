# 實驗重現操作指南

本專案有兩個主要實驗腳本：

| 腳本 | 用途 |
|---|---|
| `examples/RPNI/run_regular_experiment.py` | Regular automata DFA search（teacher 是 automata 資料夾裡的 DFA） |
| `examples/RPNI/run_realworld_experiment.py` | Real-world dataset DFA search（teacher 是已訓練好的 RNN classifier） |

---

## 1. 可用的 language / dataset 名稱

三個腳本都支援 `--languages name1,name2`。

- **Regular**: `SecureHandshake`、`DocumentReleaseWorkflow`、`MultiObligationOrder`
- **Real-world**: `mnist`、`ECG`、`wafer`

---

## 2. 重現論文的實驗

其他參數已經在 `run_regular_experiment.py` / `run_realworld_experiment.py` 的 `DEFAULT_LANGUAGE_CONFIGS` 裡寫好，不用另外改，跑的時候只要蓋掉下面這幾個就好。

**Regular**：

```bash
python examples/RPNI/run_regular_experiment.py --agreement_threshold 0.8 --batch_size 1000 --max_evaluations 3000
```

**Real-world**：

```bash
python examples/RPNI/run_realworld_experiment.py --agreement_threshold 0.8 --batch_size 1000 --max_evaluations 3000
```

**注意：上面兩個指令都只跑每個任務固定的單一 test instance，不會取平均、沒有變異數**——`DEFAULT_LANGUAGE_CONFIGS` 裡每個任務都寫了一條 `test_instance`，這條指令沒有蓋掉它，所以每個任務都是單次結果。論文表格裡每一格因此是單一 instance 的單次數字，不是多次重複的平均。

如果要改成每個任務隨機產生多條 test instance、各跑一次：

```bash
python examples/RPNI/run_regular_experiment.py --agreement_threshold 0.8 --batch_size 1000 --max_evaluations 3000 --num_test_instances 10
```

加上 `--num_test_instances N` 會改成隨機產生 N 條序列（regular 是從 teacher DFA 隨機走出來，real-world 是取 training set 前 N 條），每條各自跑一次 beam/SA/GA/PSO。注意跑的時間會直接乘上 N 倍。

`experiment_log.txt` 預設只印出每個 instance 各自的表格，不會自動平均。`src/experiments/runner.py` 提供 `print_averaged_summary(results)`，吃跟 `print_suite_summary` 一樣格式的 dict（`{f"{task}_instance_{idx:02d}": run_search_suite(...) 的回傳值}`），印出每個任務跨 instance 的 mean±std；`from experiments.runner import print_averaged_summary` 後直接呼叫即可。

---

## 3. 命令列可覆蓋的參數

| 參數 | 說明 |
|---|---|
| `--agreement_threshold` | 最終 DFA 需要達到的最低 training agreement（與 black box teacher 的一致率）門檻 |
| `--delta` | KL-LUCB 信心參數（failure probability），越小代表信心界越保守 |
| `--tau` | KL-LUCB 收斂精度（agreement 估計的容忍誤差），越小越精確但要更多樣本 |
| `--batch_size` | 每輪抽樣／評估用的樣本數 |
| `--beam_size` | Beam search 每輪保留的候選 DFA 數量 |
| `--init_num_samples` | 建立初始 DFA 時使用的樣本數 |
| `--edit_distance` | 產生候選／擾動樣本時允許的最大 Levenshtein 編輯距離 |
| `--max_length` | 產生測試序列的最大長度（僅 regular、KL-LUCB 支援） |
| `--max_evaluations` | 整個搜尋過程 agreement evaluation 次數上限（僅 regular、real-world 支援） |
| `--num_test_instances` | 要重複跑的測試 instance 數量（僅 regular、real-world 支援） |
| `--parallel` / `--no_parallel` | 是否用多執行緒平行做 KL-LUCB 抽樣與 agreement 評估  |
| `--n_jobs` | 平行模式下使用的 worker 執行緒數 |
| `--num_seeds` | 僅 KL-LUCB：要跑幾個 random seed 並取平均／標準差，預設 10 |

### 實驗中限制初始 DFA 狀態數

Beam / SA / GA / PSO 的搜尋方法本身不要求初始 DFA 落在特定狀態數範圍——狀態數多少都能跑。但為了讓實驗結果之間可以互相比較（每個任務都看得出「狀態數隨搜尋逐步下降」的趨勢），程式刻意加了一道篩選：只有初始 DFA 落在 `init_state_range=(25, 65)`（`AutomataBeamSearch.automata_beam()` 裡的固定值，目前沒有開放 CLI 覆蓋）才會拿來繼續跑；不在這個範圍內就捨棄並重新抽樣，最多重試 `max_init_attempts=40` 次，40 次都不在範圍內的話，這筆實驗會直接失敗跳過（log 會印 `[ERROR] Initial DFA construction failed`，狀態會被標成 `[SKIPPED]`）。

`--init_num_samples` 決定建初始 DFA 時觀察幾條樣本路徑，會直接影響初始 DFA 的狀態數。如果某個任務被跳過，可檢查 `--init_num_samples` 是不是相對這個任務的字母表/edit_distance 設太小。

---

## 4. 結果欄位

**Regular / Real-world**：純文字 log，在 `test_result/{regular,realworld}_{threshold}_{batch}/experiment_log.txt`。每個任務一段，先印任務標頭（regular 是 `teacher_states`/`initial_states`；real-world 是 `clf_train`/`clf_test`/`clf_test_novel`/`initial_states`），再印一張表：

| 欄位 | 說明 |
|---|---|
| `Method` | BeamSearch / SA / GA / PSO |
| `Train (Init→Final)` | 初始／refined DFA 的 training agreement（與 black box teacher 的一致率），末尾 ✓/✗ 表示有沒有達到 `--agreement_threshold` |
| `Validation (Init→Final)` | 初始／refined DFA 的 validation agreement（與初始 DFA 建構樣本的一致率） |
| `States` | final DFA 的 state 數 |
| `Time(s)` | 執行時間（秒） |

如果要機器可讀的 CSV，用 `python analysis/parse_results.py` 把 `test_result/` 底下所有 `experiment_log.txt` 彙整成 `analysis/summary_table.csv`（欄位：`config`、`domain`、`threshold`、`batch_size`、`automaton`、`teacher_states`、`initial_states`、`clf_train`、`clf_test`、`method`、`train_init`、`train_final`、`val_init`、`val_final`、`final_states`、`time_s`），細節見 [`README.md`](README.md#結果整理與畫圖)。

---

## 5. 確認結果趨勢

對照 `test_result/regular_0.8_1000/`、`test_result/realworld_0.8_1000/`、`test_result/regular_0.9_1000/`、`test_result/realworld_0.9_1000/` 裡的 `experiment_log.txt` 最下方表格。

本專案多浮點數運算，不同機器跑出來的數字不會逐位元相同，改看下面這幾個趨勢是否一致：

1. beam 的 time 大部分最小（12 個任務裡 8 個 beam 最快，其餘 4 個被 SA/GA/PSO 些微超前）
2. beam 通常能找到減少 state、且 agreement 符合門檻的解；若沒有任何候選達到門檻，beam 會回傳 training agreement 最大的候選作為最終解
3. 門檻從 0.8 調到 0.9 時，regular tasks 通常會找到 states 較大，但 agreement 達門檻的候選；而 real-world tasks 因沒有任何候選達到門檻，beam 會回傳 training agreement 最大的候選作為最終解
4. 相較於 baseline，beam 通常能用較少或相同的 state 數達到門檻，但不是每個任務都同時贏 states 和 agreement

---

最後更新：2026-09-10
