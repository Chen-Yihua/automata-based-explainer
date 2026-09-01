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

---

## 3. 命令列可覆蓋的參數

| 參數 | 說明 |
|---|---|
| `--agreement_threshold` | 最終 DFA 需要達到的最低 agreement（與 teacher 的一致率）門檻 |
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

---

## 4. 結果欄位

**Regular / Real-world**：log 在 `test_result/{regular,realworld}_{threshold}_{batch}/experiment_log.txt`，`results.csv` 主要欄位：

| 欄位 | 說明 |
|---|---|
| `method` | beam / sa / ga / pso |
| `initial_train_acc` / `final_train_acc` | 初始／refined DFA 的 training agreement |
| `initial_validation_acc` / `final_validation_acc` | 初始／refined DFA 的 validation agreement |
| `states` | final DFA 的 state 數 |
| `time_s` | 執行時間（秒） |
| `success` | 是否達到 agreement threshold |

Regular 另有 `teacher_train_acc`、`teacher_test_acc`、`teacher_states`；Real-world 另有 `clf_train_acc`、`clf_test_acc`、`init_states`。

---

## 5. 確認結果趨勢

本專案多浮點數運算，不同機器跑出來的數字不會逐位元相同，比對重現結果時看以下趨勢：

1. beam 的 `time` 大部分最小
2. beam 整體上能找到 `state` 最少、`agreement` 最高的解
3. 除了 wafer 任務，其他任務能達 agreement threshold 0.8

---

最後更新：2026-09-01
