# 這個資料夾在本專案裡的用法

`external_modules/Explaining-FA/` 是外部套件（Automata Explanation Tool，見
`README.MD`），整個資料夾在 `.gitignore` 裡被排除，只有下面這兩個檔案有
force-add 進版控，其餘（`main.py`、`dpi_explain.py`、`solve_maze.py`、
`corpus_fa_accepted/`、`corpus_fa_rejected/`、`dna/` 等）是上游自帶的 CLI
工具跟範例語料，本專案沒有用到，不需要進版控。

## 實際用到的檔案

- `language/__init__.py`
- `language/explain.py`

`src/learner/dfa_learner.py` 在 DELTA 操作裡用
`importlib.import_module("language.explain").Language` 動態載入這個模組
（先把 `external_modules/Explaining-FA` 加進 `sys.path`），用來算 DELTA
操作需要的 contrastive explanation（CXp）。`explain.py` 本身只 import 外部
套件（`libmata`、`pysat`）跟標準庫，沒有再 import 這個資料夾裡其他檔案。

## 依賴

`language/explain.py` 需要 `libmata`、`python-sat` 這兩個套件，已經列在
專案根目錄的 `requirements.txt` 裡。
