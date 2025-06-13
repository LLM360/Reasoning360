import requests
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus
from itertools import cycle
import threading
# import os
# import random
from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS

# List of available sandbox URLs
SANDBOX_URLS = [
    "http://fs-mbz-gpu-959:8080/run_code",
    "http://fs-mbz-gpu-954:8080/run_code",
]

# Create a thread-safe cycle iterator for round-robin
url_cycle = cycle(SANDBOX_URLS)
url_lock = threading.Lock()

def get_next_url():
    """Get the next URL in round-robin fashion thread-safely."""
    with url_lock:
        return next(url_cycle)

# def get_next_url():
    # """Get a random URL from the sandbox pool (lock-free)."""
    # return random.choice(SANDBOX_URLS)

def code_exec_sandboxfusion(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS):
    """
    Execute Python code using SandboxFusion remote service with load balancing.
    
    Args:
        code: Python code to execute
        stdin: Optional input to pass to the code (currently not supported)
        timeout: Timeout in seconds (default from utils)
        
    Returns:
        tuple: (success: bool, output: str)
    """
    request_data = {
        "language": "python",
        "code": code,
        "stdin": stdin,
        "run_timeout": timeout
    }

    
    # Try each URL in case of failure
    for _ in range(len(SANDBOX_URLS)):
        try:
            url = get_next_url()
            response = requests.post(url, json=request_data, timeout=timeout+2)
            
            if response.status_code == 200:
                result = RunCodeResponse(**response.json())
                if result.status == RunStatus.Success:
                    return True, result.run_result.stdout
                else:
                    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{result.run_result.stdout}\n\nSTDERR:\n{result.run_result.stderr}"
                    
        except requests.exceptions.RequestException as e:
            # If this URL fails, we'll try the next one
            continue
            
    # If we get here, all URLs failed
    return False, _ERROR_MSG_PREFIX + "All sandbox instances failed to process the request"

def code_exec_sandboxfusion_with_pytest(code, pytest_code, timeout=_DEFAULT_TIMEOUT_SECONDS):
    """
    Execute Python code with pytest using SandboxFusion remote service.
    
    Args:
        code: Python solution code
        pytest_code: Pytest test code
        timeout: Timeout in seconds
        
    Returns:
        tuple: (success: bool, output: str)
    """
    # Combine the solution code and test code
    combined_code = f"""
{code}

{pytest_code}
"""
    return code_exec_sandboxfusion(combined_code, timeout=timeout)