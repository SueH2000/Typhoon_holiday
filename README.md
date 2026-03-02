# Typhoon Holiday App（Render/Railway 可部署版）

這份專案現在有兩種模式：

1. `MODEL_MODE=ml`：使用你的模型檔（正式模式）
2. `MODEL_MODE=rule`：不需要模型檔，先用規則跑流程（展示模式）

---

## 1) 先確認「模型和腳本」有沒有接好

### 為什麼要先做這步？（初學者版）
部署失敗最常見原因不是程式語法，而是：
- 程式找不到模型檔
- 檔名對了但路徑不對
- 環境變數沒設定

所以先做「連結檢查」，你會更快定位問題。

### 你的程式現在怎麼連模型
`predict_service.py` 會讀三個路徑：
- `KNN_IMPUTER_PATH`（預設 `models/kNN_imputer.joblib`）
- `MINMAX_SCALER_PATH`（預設 `models/MMscaler.joblib`）
- `MODEL_PATH`（預設 `models/rf_model.joblib`）

如果你沒設定環境變數，就用 `models/` 資料夾內的預設檔名。

### 本機檢查指令
```bash
python scripts/check_model_link.py  # 可在專案根目錄直接執行
# 或：python predict_service.py  # 內建自我檢查
```

你會看到每個檔案是 `[OK]` 還是 `[MISSING]`。

### API 檢查（部署後也可用）
```bash
GET /app/model-status
```

這個端點會回傳：
- mode（目前是 ml 還是 rule）
- 三個模型檔實際路徑
- 每個檔案是否存在
- 有沒有載入錯誤

---


### 如果你在 Codespaces 安裝套件時遇到 `ResolutionImpossible`
這通常是「版本鎖太死」或「網路/鏡像暫時抓不到套件」造成。

建議排查順序（初學者好記版）：
1. 先更新 pip：`python -m pip install --upgrade pip`
2. 用範圍版本安裝：`python -m pip install -r requirements.txt`
3. 若你只想先跑網站流程，可先安裝最小集合：
   `python -m pip install Flask gunicorn requests`
4. 最後再補 ML 套件（pandas/numpy/scikit-learn/joblib）。

> 觀念：先讓服務跑起來，再補模型依賴，除錯會快很多。

## 2) 本機啟動

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

開 `http://localhost:5000`。

---

## 3) 部署到 Render（免費方案）

## Step A. 把程式上傳到 GitHub
```bash
git add .
git commit -m "prepare render deployment"
git push origin <你的分支或main>
```

## Step B. Render 建立服務
1. 登入 Render → **New + Web Service**
2. 選你的 GitHub repo
3. Render 會讀 `render.yaml`（已幫你寫好）
4. 確認：
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

## Step C. 設定環境變數（重點）
在 Render 的 Environment 設定：

### 方案 1（先跑展示）
- `MODEL_MODE=rule`

優點：不需要先上傳模型，先確保網站能用。

### 方案 2（正式模型）
- `MODEL_MODE=ml`
- `KNN_IMPUTER_PATH=models/kNN_imputer.joblib`
- `MINMAX_SCALER_PATH=models/MMscaler.joblib`
- `MODEL_PATH=models/rf_model.joblib`

接著你要確保三個模型檔真的在 repo 的 `models/` 裡，且有 push 到 GitHub。

> 如果模型很大不適合放 GitHub，建議先用 `MODEL_MODE=rule` 上線流程，
> 之後再改成外部檔案儲存（例如 S3）+ 啟動時下載。

## Step D. 部署後驗收
拿到網址後依序測：
1. `/health`
2. `/app/model-status`
3. `/app/counties`
4. `/app/locations?county=臺北市`
5. `/app/predict?county=臺北市&locationName=信義區`

---

## 4) 手機使用方式

1. 手機開 Render 網址
2. iPhone(Safari) → 分享 → 加入主畫面
3. Android(Chrome) → 選單 → 加入主畫面

---


## 6) 可以在 GitHub Codespaces 測試嗎？可以，而且很適合初學者

可以，**非常建議**。

### 為什麼 Codespaces 很適合你現在的階段
- 不用先在自己電腦安裝一堆版本（Python/套件衝突會少很多）。
- 測試環境在雲端，跟部署思維更接近。
- 你可以直接把同一份程式推到 GitHub，再接 Render。

### Codespaces 實作步驟
1. 到 GitHub repo 頁面，按 **Code → Codespaces → Create codespace on main**。
2. 開終端機後執行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. 先檢查模型路徑（這一步可以快速抓出 80% 的部署問題）：

```bash
python scripts/check_model_link.py  # 可在專案根目錄直接執行
# 或：python predict_service.py  # 內建自我檢查
```

4. 啟動服務：

```bash
MODEL_MODE=rule python app.py
```

5. 在 Codespaces 的 **Ports** 面板把 `5000` 設為 Public，點開瀏覽器。
6. 驗收 API：
   - `/health`
   - `/app/model-status`
   - `/app/predict?county=臺北市&locationName=信義區`

### 如果你要在 Codespaces 測 `MODEL_MODE=ml`
- 你需要把三個模型檔放到 repo（或下載到 `models/`）
- 並設定：

```bash
export MODEL_MODE=ml
export KNN_IMPUTER_PATH=models/kNN_imputer.joblib
export MINMAX_SCALER_PATH=models/MMscaler.joblib
export MODEL_PATH=models/rf_model.joblib
python app.py
```

初學者建議流程：**先 rule 成功，再切 ml**，每次只改一件事，最不容易卡住。

---

## 5) 程式檔案分工（幫你建立維護習慣）
- `app.py`：路由/API 與主流程控制
- `predict_service.py`：模型路徑、載入、推論邏輯
- `scripts/check_model_link.py`：部署前後快速檢查模型連結
- `templates/index.html` + `static/style.css`：手機前端

把「模型邏輯」和「Web 邏輯」拆開，是很重要的工程習慣。
