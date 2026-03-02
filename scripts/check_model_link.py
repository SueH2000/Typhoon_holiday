from predict_service import inspect_model_status

if __name__ == '__main__':
    status = inspect_model_status()
    print('=== 模型連結檢查 ===')
    for key, path in status['paths'].items():
        ok = 'OK' if status['exists'][key] else 'MISSING'
        print(f'- {key}: {path} [{ok}]')
    print(f"ready: {status['ready']}")
