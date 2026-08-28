# 資料集來源

本目錄包含三項真實世界任務（MNIST 筆畫序列、ECG5000、Wafer）所使用的原始資料，均取自公開資料集。以下記錄各資料集的來源、授權與本專案實際使用的檔案。

## MNIST 筆畫序列（`mnist-digits-as-stroke-sequences/`）

- **來源**：[edwin-de-jong/mnist-digits-stroke-sequence-data](https://github.com/edwin-de-jong/mnist-digits-stroke-sequence-data/wiki/MNIST-digits-stroke-sequence-data)
- **內容**：將 MNIST 手寫數字圖片轉換為筆畫方向序列（上、下、左、右）的資料集。
- **本專案使用的檔案**：僅 `mnist_strokes.pkl`（由 [`dataset/mnist_stroke_loader.py`](mnist_stroke_loader.py) 讀取）。原始來源提供的是一整套用來從 MNIST 圖片產生筆畫序列的 C++/CMake 建置工具；本目錄只保留轉換完成的 `.pkl` 結果，未包含該建置工具本身與中間產物。若需要重新產生 `mnist_strokes.pkl`（例如想調整轉換參數），請至上方連結另行取得原始工具。

## Wafer Sensors（`Wafer/`）

- **來源**：[UCR/UEA Time Series Classification Archive — Wafer](https://www.timeseriesclassification.com/description.php?Dataset=Wafer)
- **內容**：半導體晶圓製程感測器量測序列，用於偵測異常晶圓。
- **本專案使用的檔案**：僅 `Wafer_TRAIN.txt`、`Wafer_TEST.txt`（由 [`dataset/Wafer_loader.py`](Wafer_loader.py) 讀取）。來源網站另提供 `.arff`、`.ts` 等格式，本專案未使用，未保留於本目錄。

## ECG5000（`ECG5000/`）

- **來源**：[UCR/UEA Time Series Classification Archive — ECG5000](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000)
- **內容**：心電圖心跳波形序列，用於偵測異常心跳。
- **本專案使用的檔案**：僅 `ECG5000_TRAIN.txt`、`ECG5000_TEST.txt`（由 [`dataset/ECG_loader.py`](ECG_loader.py) 讀取）。來源網站另提供 `.arff`、`.ts` 等格式，本專案未使用，未保留於本目錄。

---

三個 loader（`ECG_loader.py`、`Wafer_loader.py`、`mnist_stroke_loader.py`）皆由 `models/*_classifier.py` 訓練腳本呼叫，將上述原始資料轉換為符號序列，作為 [`RUNNING.md`](../RUNNING.md) 中 real-world 任務的黑盒 teacher（RNN 分類器）訓練資料。
