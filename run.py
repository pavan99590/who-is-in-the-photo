"""Start the app:  python run.py   (opens your browser; fully offline)"""
import argparse
import threading
import webbrowser

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    if not a.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Open {url}  (Ctrl+C to stop). Everything stays on this computer.")
    uvicorn.run("faceid.server:app", host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()

