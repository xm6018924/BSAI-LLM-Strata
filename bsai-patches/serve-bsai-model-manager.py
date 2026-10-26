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
STRATA_PORT = 8080   # BSAI Strata 引擎 / BSAI 网页（切换模型时原地重启，保持占用）
FLINT_PORT = 8082    # Flint 40B 独立 llama.cpp 引擎（与 Strata 并行，互不占用端口）
INSTALL_DIR = Path(__file__).parent
PYTHON_EXE = sys.executable

# Model options
MODELS = {
    "Q2_0": {"json": "strata-q2_0.json", "desc": "最快 / fastest"},
    "IQ2_XS": {"json": "strata-iq2_xs.json", "desc": "略好 / better"},
    "IQ3_XXS": {"json": "strata-iq3_xxs.json", "desc": "更好 / good"},
    "IQ3_S": {"json": "strata-iq3_s.json", "desc": "最好质量 / best quality"},
    "flint": {"json": None, "desc": "Flint 40B 编程模型 / coding"},
}

# Flint 40B: 独立 llama.cpp 引擎（非 Strata），固定端口 FLINT_PORT=8082，不占用 8080
FLINT_MODEL_CANDIDATES = ("flint-UD-Q3_K_XL.gguf", "flint-UD-Q4_K_XL.gguf")
FLINT_ALIAS = "Flint-40B"


def _integration_root():
    """整合包根（绘世启动器.exe 同级）——相对路径探测，跨盘符可移植。"""
    p = Path(__file__).resolve().parents[4]
    return p


def _find_llama_server():
    """在整合包根下寻找移植的 llama-server.exe（优先 BSAI-llama.cpp）。"""
    root = _integration_root()
    for rel in ("BSAI-llama.cpp", "llama.cpp"):
        cand = root / rel / "llama-server.exe"
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "llama-server.exe 未找到（整合包根/BSAI-llama.cpp 或 /llama.cpp）: %s" % root)


def _find_flint_model():
    """在 models/LLM/flint/ 下寻找 Flint GGUF（优先 Q3_K_XL）。"""
    try:
        import folder_paths
        base = Path(folder_paths.models_dir)
    except Exception:
        base = _integration_root() / "ComfyUI" / "models"
    for name in FLINT_MODEL_CANDIDATES:
        cand = base / "LLM" / "flint" / name
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "Flint GGUF 未找到（%s 下 %s）" % (base / "LLM" / "flint",
                                     " / ".join(FLINT_MODEL_CANDIDATES)))

strata_process = None   # 跑在 8080 的 Strata 引擎
flint_process = None     # 跑在 8082 的 Flint llama.cpp 引擎（独立进程）
current_model = None
lock = threading.Lock()


def start_strata(model_name):
    """Start engine. Strata 固定占用 8080；Flint 独立占用 8082，两者并行互不杀进程。"""
    global strata_process, flint_process, current_model

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    # --- Flint 40B：独立 llama.cpp 引擎，端口 8082，不影响 8080 上的 Strata ---
    if model_name == "flint":
        with lock:
            # 只重启旧的 Flint 进程；Strata 引擎保持运行
            if flint_process:
                try:
                    # Kill entire process tree on Windows
                    subprocess.run(["taskkill", "/f", "/t", "/pid", str(flint_process.pid)],
                                 capture_output=True)
                    flint_process.wait(timeout=5)
                except:
                    try:
                        flint_process.kill()
                    except:
                        pass
            time.sleep(2)

            exe = _find_llama_server()
            model = _find_flint_model()
            args = [exe, "-m", model, "--host", "127.0.0.1",
                    "--port", str(FLINT_PORT),
                    "-c", "65536", "-fa", "auto", "--jinja", "--parallel", "1",
                    "--alias", FLINT_ALIAS, "--temp", "0.6", "--top-p", "0.95",
                    "--top-k", "20", "--repeat-penalty", "1.05", "-ngl", "99",
                    "-n", "65536"]
            flint_process = subprocess.Popen(
                args,
                cwd=str(Path(exe).parent),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print(f"[BSAI Manager] Started Flint 40B (llama.cpp) on port {FLINT_PORT}: {model}")
        wait_for_engine(FLINT_PORT)
        current_model = model_name
        print(f"[BSAI Manager] Model flint is now active on port {FLINT_PORT}!")
        return

    # --- Strata 模型：原地重启 8080 上的 Strata 引擎；Flint(8082) 不受影响 ---
    with lock:
        # Stop existing Strata process (kill entire process tree)
        if strata_process:
            try:
                # Kill entire process tree on Windows
                subprocess.run(["taskkill", "/f", "/t", "/pid", str(strata_process.pid)],
                             capture_output=True)
                strata_process.wait(timeout=5)
            except:
                try:
                    strata_process.kill()
                except:
                    pass

        # Also kill any remaining strata.exe processes
        try:
            subprocess.run(["taskkill", "/f", "/im", "strata.exe"], capture_output=True)
        except:
            pass

        # Wait a bit for ports and GPU to release
        time.sleep(3)

        json_file = INSTALL_DIR / MODELS[model_name]["json"]
        if not json_file.exists():
            raise FileNotFoundError(f"Config file not found: {json_file}")

        server_py = INSTALL_DIR / "serve" / "server.py"

        strata_process = subprocess.Popen(
            [PYTHON_EXE, str(server_py), "--engine", "strata", "--config", str(json_file), "--port", str(STRATA_PORT)],
            cwd=str(INSTALL_DIR),
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        print(f"[BSAI Manager] Started Strata with model: {model_name}")

    wait_for_engine(STRATA_PORT)

    # Only set current_model after engine is fully ready
    current_model = model_name
    print(f"[BSAI Manager] Model {model_name} is now active!")


def wait_for_engine(port, timeout=600):
    """Wait for engine on given port to be ready"""
    import urllib.request
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
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
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        target_model = qs.get("model", [""])[0]
        # Flint -> 等 8082；其余（Strata）-> 等 8080
        wait_port = FLINT_PORT if target_model == "flint" else STRATA_PORT
        if parsed.path == "/" or parsed.path == "/wait":
            # Waiting page: auto-redirect when target engine is ready
            check_url = f"http://127.0.0.1:{wait_port}/v1/models"
            redirect_url = f"http://127.0.0.1:{wait_port}/"
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
        const r = await fetch("__CHECK_URL__", {mode: "no-cors"});
        window.location.href = "__REDIRECT_URL__";
      } catch (e) {
        setTimeout(checkReady, 2000);
      }
    }
    checkReady();
  </script>
</body>
</html>'''
            html = html.replace("__CHECK_URL__", check_url).replace("__REDIRECT_URL__", redirect_url)
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
    
    # Get initial model from command line
    initial_model = sys.argv[1] if len(sys.argv) > 1 else "IQ3_XXS"
    
    print(f"[BSAI Manager] Starting with model: {initial_model}")
    print(f"[BSAI Manager] Waiting page: http://127.0.0.1:{MANAGER_PORT}/")
    print(f"[BSAI Manager] Web UI will open automatically when ready")
    
    # Start HTTP manager server first (so waiting page is available)
    httpd = socketserver.TCPServer(("127.0.0.1", MANAGER_PORT), ManagerHandler)
    manager_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    manager_thread.start()
    
    # Open waiting page immediately（带目标模型参数：Flint 会跳 8082，Strata 跳 8080）
    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{MANAGER_PORT}/wait?model={initial_model}")
    
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
