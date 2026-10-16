"""
BSAI Strata Model Manager
- Manages Strata engine lifecycle
- Provides HTTP API for model switching
"""
import os
import sys
import json
import time
import subprocess
import threading
import http.server
import socketserver
from pathlib import Path

# Config
MANAGER_PORT = 8081
STRATA_PORT = 8080
INSTALL_DIR = Path(__file__).parent
PYTHON_EXE = sys.executable

# Model options
MODELS = {
    "Q2_0": {"json": "strata-q2_0.json", "desc": "最快 / fastest"},
    "IQ2_XS": {"json": "strata-iq2_xs.json", "desc": "略好 / better"},
    "IQ3_XXS": {"json": "strata-iq3_xxs.json", "desc": "更好 / good"},
    "IQ3_S": {"json": "strata-iq3_s.json", "desc": "最好质量 / best quality"},
}

current_process = None
current_model = None
lock = threading.Lock()


def start_strata(model_name):
    """Start Strata engine with given model"""
    global current_process, current_model
    
    with lock:
        # Stop existing process (kill entire process tree)
        if current_process:
            try:
                # Kill entire process tree on Windows
                subprocess.run(["taskkill", "/f", "/t", "/pid", str(current_process.pid)], 
                             capture_output=True)
                current_process.wait(timeout=5)
            except:
                try:
                    current_process.kill()
                except:
                    pass
        
        # Also kill any remaining strata.exe processes
        try:
            subprocess.run(["taskkill", "/f", "/im", "strata.exe"], capture_output=True)
        except:
            pass
        
        # Wait a bit for ports and GPU to release
        time.sleep(3)
        
        # Start new process
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        json_file = INSTALL_DIR / MODELS[model_name]["json"]
        if not json_file.exists():
            raise FileNotFoundError(f"Config file not found: {json_file}")
        
        server_py = INSTALL_DIR / "serve" / "server.py"
        
        current_process = subprocess.Popen(
            [PYTHON_EXE, str(server_py), "--engine", "strata", "--config", str(json_file), "--port", str(STRATA_PORT)],
            cwd=str(INSTALL_DIR),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print(f"[BSAI Manager] Started Strata with model: {model_name}")
        
        # Wait for engine to be ready
        wait_for_engine()
        
        # Only set current_model after engine is fully ready
        current_model = model_name
        print(f"[BSAI Manager] Model {model_name} is now active!")


def wait_for_engine(timeout=600):
    """Wait for Strata engine to be ready"""
    import urllib.request
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{STRATA_PORT}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[BSAI Manager] Engine ready!")
                    return True
        except:
            pass
        time.sleep(2)
    print(f"[BSAI Manager] Engine wait timeout!")
    return False


class ManagerHandler(http.server.BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def send_html(self, html, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))
    
    def do_GET(self):
        if self.path == "/" or self.path == "/wait":
            # Waiting page: auto-redirect when 8080 is ready
            html = '''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>BSAI Strata - Loading</title>
  <style>
    body { background: #1a1a2e; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; font-family: Arial, sans-serif; }
    .logo { width: 96px; height: 96px; margin-bottom: 20px; border-radius: 50%; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
    .title { font-size: 24px; margin-bottom: 10px; }
    .sub { font-size: 16px; color: #888; }
  </style>
</head>
<body>
  <img src="/logo.png" class="logo" alt="BSAI Logo">
  <div class="title">BSAI Strata 正在加载...</div>
  <div class="sub">首次启动需要 1-2 分钟，请耐心等待，页面会自动跳转</div>
  <script>
    async function checkReady() {
      try {
        const r = await fetch("http://127.0.0.1:8080/v1/models", {mode: "no-cors"});
        window.location.href = "http://127.0.0.1:8080/";
      } catch (e) {
        setTimeout(checkReady, 2000);
      }
    }
    checkReady();
  </script>
</body>
</html>'''
            self.send_html(html)
        elif self.path == "/logo.png":
            # Serve BSAI logo
            logo_path = INSTALL_DIR / "serve" / "web" / "bsai-logo.png"
            if not logo_path.exists():
                logo_path = INSTALL_DIR / "bsai-patches" / "serve-web-bsai-logo.png"
            if logo_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(logo_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
        elif self.path == "/api/models":
            self.send_json({
                "current": current_model,
                "models": MODELS
            })
        elif self.path == "/api/current":
            self.send_json({"current": current_model})
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        if self.path == "/api/switch":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            model = data.get("model")
            try:
                # Start in background thread
                threading.Thread(target=start_strata, args=(model,), daemon=True).start()
                self.send_json({"status": "switching", "model": model})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)
        else:
            self.send_json({"error": "Not found"}, 404)
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
    
    def log_message(self, format, *args):
        # Suppress logs
        pass


def main():
    global current_model
    
    # ===== Auto kill old process occupying the manager port =====
    print(f"[BSAI Manager] Checking port {MANAGER_PORT} for old processes...")
    try:
        import subprocess
        # Find PID occupying the port
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{MANAGER_PORT}" in line and "LISTENING" in line:
                parts = line.split()
                old_pid = parts[-1]
                if old_pid and old_pid.isdigit():
                    print(f"[BSAI Manager] Killing old process PID: {old_pid}")
                    subprocess.run(["taskkill", "/f", "/pid", old_pid], capture_output=True)
                    time.sleep(1)
    except Exception as e:
        print(f"[BSAI Manager] Port check skipped: {e}")
    
    # Get initial model from command line
    initial_model = sys.argv[1] if len(sys.argv) > 1 else "IQ3_XXS"
    
    print(f"[BSAI Manager] Starting with model: {initial_model}")
    print(f"[BSAI Manager] Waiting page: http://127.0.0.1:{MANAGER_PORT}/")
    print(f"[BSAI Manager] Web UI will open automatically when ready")
    
    # Start HTTP manager server first (so waiting page is available)
    httpd = socketserver.TCPServer(("127.0.0.1", MANAGER_PORT), ManagerHandler)
    manager_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    manager_thread.start()
    
    # Open waiting page immediately
    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{MANAGER_PORT}/")
    
    # Start Strata engine in background
    threading.Thread(target=start_strata, args=(initial_model,), daemon=True).start()
    
    print(f"[BSAI Manager] Ready! Waiting for engine to load...")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
