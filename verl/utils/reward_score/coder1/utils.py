import requests

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30   # 30 seconds is the default timeout for the executor


def check_executor_alive(executor):
    try:
        return requests.get(executor + "/").status_code in [200, 404]
    except Exception:
        return False
