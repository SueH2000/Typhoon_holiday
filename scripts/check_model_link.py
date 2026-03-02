
from __future__ import annotations

import sys
from pathlib import Path

# 初學者說明：
# 你用 `python scripts/check_model_link.py` 執行時，Python 只會先看 scripts/ 資料夾。
# 但 predict_service.py 在專案根目錄，所以要把「專案根目錄」加進匯入路徑。
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from predict_service import inspect_model_status

if __name__ == '__main__':
    status = inspect_model_status()
    print('=== 模型連結檢查 ===')
    for key, path in status['paths'].items():
        ok = 'OK' if status['exists'][key] else 'MISSING'
        print(f'- {key}: {path} [{ok}]')
    print(f"ready: {status['ready']}")
