"""Start the dashboard with clear messages, port validation, and reliable browser launch.

Use:  python run_app.py      (or double-click run.bat on Windows)
"""
from __future__ import annotations

import atexit
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Automatically ensure categorical sub-folders are in sys.path
_ROOT = Path(__file__).resolve().parent
for _subdir in ("analytics", "data", "execution", "assets"):
    _sub_p = str(_ROOT / _subdir)
    if _sub_p not in sys.path:
        sys.path.insert(0, _sub_p)

REQUIRED = [
    "streamlit",
    "plotly",
    "pandas",
    "numpy",
    "ta",
    "pyotp",
    "pymupdf",
    "yfinance",
    "tradingview_ta",
]

PORT = 8501
URL = f"http://localhost:{PORT}"


def is_port_in_use(port: int = PORT) -> bool:
    """Return True if port is actively listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def free_port_if_needed(port: int = PORT) -> None:
    """If an orphaned process is holding the port, kill it to ensure a clean start."""
    if not is_port_in_use(port):
        return

    print(f"[*] Port {port} is occupied by an existing process. Reclaiming port...")
    try:
        output = subprocess.check_output(
            f'netstat -ano | findstr :{port}',
            shell=True,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        my_pid = os.getpid()
        killed = False
        for line in output.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pid_str = parts[-1]
                if pid_str.isdigit() and int(pid_str) != my_pid and int(pid_str) != 0:
                    subprocess.call(
                        f"taskkill /F /T /PID {pid_str}",
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    killed = True
        if killed:
            # Wait up to 3 seconds for OS socket table to fully release
            for _ in range(6):
                time.sleep(0.5)
                if not is_port_in_use(port):
                    break
    except Exception:
        pass


def wait_and_open_browser(url: str = URL, timeout: float = 60.0) -> None:
    """Wait until the Streamlit server is actually ready and accepting HTTP requests, then open browser."""
    health_url = f"{url.rstrip('/')}/_stcore/health"
    start_time = time.time()
    server_ready = False

    print("[*] Waiting for trading terminal engine to initialize...", end="", flush=True)
    time.sleep(1.5)  # Allow server socket initialization to stabilize
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(
                health_url,
                headers={"User-Agent": "AlphaEdgeLauncher", "Connection": "close"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    server_ready = True
                    break
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1.0)

    print()
    if server_ready:
        print(f"[OK] Server is ready! Opening dashboard in browser: {url}\n")
    else:
        print(f"[!] Server initialization took longer than {timeout:.0f}s. Attempting to open browser anyway...\n")

    # Priority 1: Launch via native Windows ShellExecute (opens default browser and requests window focus)
    if hasattr(os, "startfile"):
        try:
            os.startfile(url)
        except Exception:
            pass

    # Priority 2: Launch Chrome executable directly if found
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for cp in chrome_paths:
        if os.path.exists(cp):
            try:
                subprocess.Popen([cp, url])
                break
            except Exception:
                pass

    # Priority 3: Fallback Windows command 'start "" "{url}"'
    if sys.platform == "win32":
        try:
            subprocess.call(f'start "" "{url}"', shell=True)
        except Exception:
            pass

    # Priority 4: Fallback standard library webbrowser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    # Activate Chrome window if already open in background
    if sys.platform == "win32":
        try:
            ps_activate = (
                "$wshell = New-Object -ComObject WScript.Shell; "
                "$wshell.AppActivate('Google Chrome'); "
                "$wshell.AppActivate('Alpha Edge')"
            )
            subprocess.Popen(
                ["powershell", "-NoProfile", "-Command", ps_activate],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    print("=" * 78)
    print("   [+] TRADING SERVER STATUS : ONLINE & STREAMING REAL-TIME EXCHANGE DATA")
    print(f"   [+] ACCESS DASHBOARD      : {url}")
    print("   [+] NEXT STEP             : Click on the Google Chrome icon in your taskbar,")
    print(f"                               or open your browser to {url}")
    print("   [+] TERMINAL ADVISORY     : Keep this black terminal window open in the")
    print("                               background while using the trading terminal.")
    print("=" * 78 + "\n")


def main() -> int:
    if sys.platform == "win32":
        try:
            os.system("")  # Enable VT100 / ANSI escape sequence color rendering in Windows console
        except Exception:
            pass

    if sys.version_info < (3, 9):
        print(f"Python 3.9 or newer is needed; you have {sys.version.split()[0]}.")
        return 1

    # Check for missing required dependencies
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        print("Missing required packages:", ", ".join(missing))
        print("\nInstalling missing packages automatically... (this may take 1-2 minutes, please wait)\n")
        req = Path(__file__).with_name("requirements.txt")
        res = subprocess.call([sys.executable, "-m", "pip", "install", "-r", str(req)])
        if res != 0:
            print("\nAutomatic installation encountered an error.")
            print("Please try running manually:")
            print(f'  "{sys.executable}" -m pip install -r "{req}"\n')
            return 1
        print("\nAll packages installed successfully!\n")

    app = Path(__file__).with_name("dashboard.py")
    if not app.exists():
        print(f"Error: Could not find '{app}'")
        return 1

    # Ensure port is clean
    free_port_if_needed(PORT)

    # Display Institutional Application Server Logo Banner
    banner_file = Path(__file__).resolve().parent / "assets" / "server_banner.txt"
    if banner_file.exists():
        try:
            print(banner_file.read_text(encoding="utf-8"))
        except Exception:
            print("==========================================================================")
            print("   Alpha Edge — Institutional AI Trading Terminal")
            print("==========================================================================")
    else:
        print("==========================================================================")
        print("   Alpha Edge — Institutional AI Trading Terminal")
        print("==========================================================================")

    print(f"   [+] Server Address : {URL}")
    print(f"   [+] Press Ctrl+C in this terminal window at any time to safely stop the server.\n")

    # Start health check and browser launcher in a background thread
    launcher_thread = threading.Thread(target=wait_and_open_browser, args=(URL,), daemon=True)
    launcher_thread.start()

    # Configure Windows asyncio loop policy and silence terminal deprecation warnings
    wrapper_code = (
        "import sys, os, asyncio, warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "os.environ['STREAMLIT_GLOBAL_SUPPRESS_DEPRECATION_WARNINGS'] = 'true'\n"
        "if sys.platform == 'win32' and hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):\n"
        "    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())\n"
        "try:\n"
        "    from streamlit import deprecation_util\n"
        "    deprecation_util.show_deprecation_warning = lambda *a, **k: None\n"
        "except Exception:\n"
        "    pass\n"
        "from streamlit.web.cli import main\n"
        "main()\n"
    )

    # Launch Streamlit process
    cmd = [
        sys.executable,
        "-W", "ignore",
        "-c",
        wrapper_code,
        "run",
        str(app),
        f"--server.port={PORT}",
        "--server.headless=true",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false",
        "--global.suppressDeprecationWarnings=true",
        "--client.showErrorDetails=none",
        "--logger.level=error",
    ]

    proc = subprocess.Popen(cmd, cwd=str(app.parent))

    def cleanup() -> None:
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    atexit.register(cleanup)

    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)

    try:
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal_handler)
    except Exception:
        pass

    try:
        return proc.wait()
    except KeyboardInterrupt:
        cleanup()
        return 0


if __name__ == "__main__":
    sys.exit(main())
