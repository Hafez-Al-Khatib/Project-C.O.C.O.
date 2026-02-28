"""Start Uvicorn with proper stdout capture."""
import subprocess
import sys

proc = subprocess.run(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--port", "8000"],
    capture_output=False,
    timeout=15,
)
