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
        # Stop existing process
        if current_process:
            try:
                current_process.terminate()
                current_process.wait(timeout=5)
            except:
                try:
                    current_process.kill()
                except:
                    pass
        
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
        current_model = model_name
        print(f"[BSAI Manager] Started Strata with model: {model_name}")
        
        # Wait for engine to be ready
        wait_for_engine()


def wait_for_engine(timeout=180):
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
    
    def do_GET(self):
        if self.path == "/api/models":
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
    
    # Get initial model from command line
    initial_model = sys.argv[1] if len(sys.argv) > 1 else "IQ3_XXS"
    
    print(f"[BSAI Manager] Starting with model: {initial_model}")
    print(f"[BSAI Manager] Manager API: http://127.0.0.1:{MANAGER_PORT}")
    print(f"[BSAI Manager] Web UI: http://127.0.0.1:{STRATA_PORT}")
    
    # Start Strata engine
    start_strata(initial_model)
    
    # Auto-open browser
    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{STRATA_PORT}/")
    
    # Start HTTP manager server
    with socketserver.TCPServer(("127.0.0.1", MANAGER_PORT), ManagerHandler) as httpd:
        print(f"[BSAI Manager] Ready! Switch models in About page.")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
