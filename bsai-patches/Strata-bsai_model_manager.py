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

def available_models():
    """动态探测本机实际可用的模型：strata-<quant>.json 存在，且 json 中
    --native / --mtp 指向的模型文件真实存在，才允许切换/启动。
    避免因缺 json 或缺模型文件导致 FileNotFoundError / 切换报错。"""
    out = {}
    for name, spec in MODELS.items():
        jp = INSTALL_DIR / spec["json"]
        if not jp.is_file():
            continue
        try:
            cfg = json.loads(jp.read_text(encoding="utf-8-sig"))
            args = cfg.get("args") or []
            ok = True
            for i, a in enumerate(args):
                if a == "--native" and i + 1 < len(args):
                    ok = ok and Path(args[i + 1]).is_file()
                elif a == "--mtp" and i + 1 < len(args):
                    ok = ok and Path(args[i + 1]).exists()
            if ok:
                out[name] = spec
        except Exception:
            continue
    return out



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
        avail = available_models()
        if model_name not in avail:
            missing = []
            if model_name in MODELS:
                jp = INSTALL_DIR / MODELS[model_name]["json"]
                if not jp.is_file():
                    missing.append(f"配置文件缺失: {jp}")
                else:
                    try:
                        cfg = json.loads(jp.read_text(encoding="utf-8-sig"))
                        args = cfg.get("args") or []
                        for i, a in enumerate(args):
                            if a == "--native" and i + 1 < len(args) and not Path(args[i + 1]).is_file():
                                missing.append(f"模型文件缺失: {args[i + 1]}")
                            elif a == "--mtp" and i + 1 < len(args) and not Path(args[i + 1]).is_file():
                                missing.append(f"MTP 文件缺失: {args[i + 1]}")
                    except Exception as e:
                        missing.append(f"配置解析失败: {e}")
            else:
                missing.append(f"未知模型: {model_name}")
            raise RuntimeError(
                f"模型 {model_name} 不可用: {'; '.join(missing)}。"
                f"当前可用模型: {', '.join(avail) if avail else '无'}。"
                f"请先双击 BSAI_Strata服务.bat 完成模型安装/配置。"
            )

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


def wait_for_engine(timeout=1800):
    """Wait for Strata engine to be ready - 30 min timeout for large models"""
    import urllib.request
    start_time = time.time()
    last_print = 0
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{STRATA_PORT}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print(f"[BSAI Manager] Engine ready!")
                    return True
        except:
            pass
        # Print progress every 30 seconds
        elapsed = int(time.time() - start_time)
        if elapsed - last_print >= 30:
            print(f"[BSAI Manager] Still loading... {elapsed}s elapsed")
            last_print = elapsed
        time.sleep(2)
    print(f"[BSAI Manager] Engine wait timeout after {timeout}s!")
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
                "models": available_models()
            })
        elif self.path == "/api/current":
            self.send_json({"current": current_model})
        elif self.path.startswith("/api/websearch"):
            # Web search using Bing China (no VPN needed)
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            if not query:
                self.send_json({"error": "No query provided"}, 400)
                return
            try:
                import urllib.request
                import re
                import ssl
                # Disable SSL verification to avoid Windows cert store issues
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                url = f"https://cn.bing.com/search?q={urllib.parse.quote(query)}&setlang=zh-CN"
                req = urllib.request.Request(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept-Language": "zh-CN,zh;q=0.9"
                })
                with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
                    html = resp.read().decode("utf-8", "replace")
                    results = []
                    # Extract Bing search results
                    blocks = re.findall(r'<li class="b_algo".*?</li>', html, re.DOTALL)
                    for block in blocks[:5]:
                        title_m = re.search(r'<h2>.*?<a.*?>(.*?)</a>', block, re.DOTALL)
                        snippet_m = re.search(r'<p class="b_lineclamp.*?">(.*?)</p>', block, re.DOTALL)
                        url_m = re.search(r'<h2>.*?<a href="(.*?)"', block, re.DOTALL)
                        if title_m:
                            title = re.sub(r'<.*?>', '', title_m.group(1)).strip()
                            snippet = re.sub(r'<.*?>', '', snippet_m.group(1)).strip() if snippet_m else ""
                            result_url = url_m.group(1) if url_m else ""
                            results.append({"title": title[:80], "snippet": snippet[:300], "url": result_url})
                    self.send_json({"query": query, "results": results})
            except Exception as e:
                self.send_json({"error": f"Search failed: {str(e)}", "results": []})
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
        elif self.path == "/api/exec":
            # Execute shell command in specified working directory
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)
            command = data.get("command", "")
            cwd = data.get("cwd", None)
            if not command:
                self.send_json({"error": "No command provided"}, 400)
                return
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                self.send_json({
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode
                })
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
            capture_output=True, text=True, encoding="gbk", errors="replace"
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
    
    # Get initial model from command line (fall back to first available)
    avail = available_models()
    initial_model = sys.argv[1] if len(sys.argv) > 1 else (next(iter(avail), None))
    if initial_model and initial_model not in avail:
        fallback = next(iter(avail), None)
        print(f"[BSAI Manager] Model {initial_model} not available, falling back to: {fallback}")
        initial_model = fallback
    
    print(f"[BSAI Manager] Starting with model: {initial_model}")
    print(f"[BSAI Manager] Waiting page: http://127.0.0.1:{MANAGER_PORT}/")
    print(f"[BSAI Manager] Web UI will open automatically when ready")
    
    # Start HTTP manager server first
    httpd = socketserver.TCPServer(("127.0.0.1", MANAGER_PORT), ManagerHandler)
    manager_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    manager_thread.start()
    
    # Auto-start model if a model is specified on command line (manual launch)
    if initial_model:
        import webbrowser
        print(f"[BSAI Manager] Auto-starting model: {initial_model}")
        threading.Thread(target=start_strata, args=(initial_model,), daemon=True).start()
        webbrowser.open(f"http://127.0.0.1:{MANAGER_PORT}/")
    else:
        print(f"[BSAI Manager] Manager service ready, no auto model load (ComfyUI internal mode)")
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
