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
from html.parser import HTMLParser
from pathlib import Path

# Config
MANAGER_PORT = 8081
STRATA_PORT = 8080   # BSAI Strata 引擎 / BSAI 网页（切换模型时原地重启，保持占用）
FLINT_PORT = 8082    # Flint 40B 独立 llama.cpp 引擎（与 Strata 并行，互不占用端口）
DEEPSEEK_PORT = 8083 # DeepSeek V4.1 独立 strata-dsv41 引擎（与 Strata 并行，互不占用端口）
INSTALL_DIR = Path(__file__).parent
PYTHON_EXE = sys.executable

# Model options
MODELS = {
    "Q2_0": {"json": "strata-q2_0.json", "desc": "最快 / fastest"},
    "IQ2_XS": {"json": "strata-iq2_xs.json", "desc": "略好 / better"},
    "IQ3_XXS": {"json": "strata-iq3_xxs.json", "desc": "更好 / good"},
    "IQ3_S": {"json": "strata-iq3_s.json", "desc": "最好质量 / best quality"},
    "flint": {"json": None, "desc": "Flint 40B 编程模型 / coding"},
    "deepseek": {"json": "strata-deepseek-v4.1-flash-ist-das-2.00bpw.json",
                 "desc": "DeepSeek V4.1 推理 / reasoning"},
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

# ------------------------------------------------------------------ BSAI 联网搜索
# 网页里的 SEARCH: 流程调用 GET /api/websearch?q=...；后端抓 Bing（国内可达，
# DuckDuckGo/Google 直连会超时），只用标准库，8 秒超时，失败就返回空结果。
SEARCH_TIMEOUT = 8
SEARCH_HOSTS = ("cn.bing.com", "www.bing.com")
SEARCH_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class _BingParser(HTMLParser):
    """从 Bing 结果页的 <li class="b_algo"> 里取 标题 / 链接 / 摘要。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results = []
        self._algo = False
        self._h2 = False
        self._cap = False
        self._href = None
        self._buf = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = (a.get("class") or "").split()
        if tag == "li" and "b_algo" in cls:
            self._algo, self._href = True, None
        elif tag == "h2" and self._algo:
            self._h2 = True
        elif tag == "a" and self._h2:
            self._href, self._buf = a.get("href"), []
        elif tag == "div" and "b_caption" in cls and self._algo:
            self._cap = True
        elif tag in ("p", "div") and self._cap:
            self._buf.append(" ")

    def handle_endtag(self, tag):
        if tag == "a" and self._h2:
            title = " ".join("".join(self._buf).split())
            if title and self._href:
                self.results.append({"title": title, "url": self._href, "snippet": ""})
            self._h2, self._href = False, None
        elif tag == "h2":
            self._h2 = False
        elif tag == "div" and self._cap:
            snip = " ".join("".join(self._buf).split())
            if snip and self.results:
                self.results[-1]["snippet"] = (self.results[-1]["snippet"] + " " + snip).strip()[:600]
            self._cap, self._buf = False, []
        elif tag == "li" and self._algo:
            self._algo = self._cap = False
            self._buf = []

    def handle_data(self, data):
        if self._h2 or self._cap:
            self._buf.append(data)


def bing_search(query, count=6):
    """Bing 抓取，无 API Key；cn.bing.com 优先，失败再试 www.bing.com。"""
    import urllib.parse
    import urllib.request
    count = max(1, min(int(count or 6), 10))
    qs = urllib.parse.urlencode({"q": query, "setlang": "zh-CN", "count": str(count + 4)})
    last = None
    for host in SEARCH_HOSTS:
        try:
            req = urllib.request.Request("https://%s/search?%s" % (host, qs), headers={
                "User-Agent": SEARCH_UA,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            })
            with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
                html = resp.read().decode("utf-8", "replace")
        except Exception as e:
            last = e
            print(f"[BSAI Manager] websearch {host} 失败: {e}")
            continue
        p = _BingParser()
        try:
            p.feed(html)
        except Exception as e:
            last = e
        out = []
        for r in p.results:
            if not r["url"].startswith("http") or any(o["url"] == r["url"] for o in out):
                continue
            out.append(r)
            if len(out) >= count:
                break
        if out:
            return out
    raise RuntimeError("搜索失败: %s" % (last or "没有解析到结果"))


strata_process = None   # 跑在 8080 的 Strata 引擎
flint_process = None     # 跑在 8082 的 Flint llama.cpp 引擎（独立进程）
deepseek_process = None  # 跑在 8083 的 DeepSeek strata-dsv41 引擎（独立进程）
current_model = None
lock = threading.Lock()


def restore_ui_patches():
    """Strata 引擎目录被更新覆盖后，启动时自动把 BSAI 工作台界面补回来。

    补丁源在插件根 bsai-patches/（strata_manager.apply_bsai_patches 同源）。
    只补 web 界面文件；serve/server.py 保留引擎更新后的版本。
    """
    patches_dir = INSTALL_DIR.parent / "bsai-patches"
    if not patches_dir.is_dir():
        return
    import shutil
    for name, rel in (("serve-web-index.html", Path("serve") / "web" / "index.html"),
                      ("serve-web-app.js", Path("serve") / "web" / "app.js"),
                      ("serve-web-bsai-logo.png", Path("serve") / "web" / "bsai-logo.png")):
        src = patches_dir / name
        dst = INSTALL_DIR / rel
        if src.is_file() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
            shutil.copy2(src, dst)
            print(f"[BSAI Manager] UI 补丁已恢复: {name}")


def start_strata(model_name):
    """Start engine. Strata 固定占用 8080；Flint 独立占用 8082；DeepSeek 独立占用 8083。"""
    global strata_process, flint_process, deepseek_process, current_model

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

    # --- DeepSeek V4.1：独立 strata-dsv41 引擎，端口 8083，停掉 8080 的 Strata 以释放显存 ---
    if model_name == "deepseek":
        with lock:
            # 停掉 8080 上的 Strata（DeepSeek 与 Qwen 同占大显存，不并行）
            if strata_process:
                try:
                    subprocess.run(["taskkill", "/f", "/t", "/pid", str(strata_process.pid)],
                                 capture_output=True)
                    strata_process.wait(timeout=5)
                except:
                    try:
                        strata_process.kill()
                    except:
                        pass
                strata_process = None
            try:
                subprocess.run(["taskkill", "/f", "/im", "strata.exe"], capture_output=True)
            except:
                pass
            # 重启旧的 DeepSeek 进程
            if deepseek_process:
                try:
                    subprocess.run(["taskkill", "/f", "/t", "/pid", str(deepseek_process.pid)],
                                 capture_output=True)
                    deepseek_process.wait(timeout=5)
                except:
                    try:
                        deepseek_process.kill()
                    except:
                        pass
            time.sleep(3)

            json_file = INSTALL_DIR / MODELS[model_name]["json"]
            if not json_file.exists():
                raise FileNotFoundError(f"Config file not found: {json_file}")

            server_py = INSTALL_DIR / "serve" / "server.py"
            deepseek_process = subprocess.Popen(
                [PYTHON_EXE, str(server_py), "--engine", "strata", "--config", str(json_file),
                 "--port", str(DEEPSEEK_PORT)],
                cwd=str(INSTALL_DIR),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            print(f"[BSAI Manager] Started DeepSeek V4.1 on port {DEEPSEEK_PORT}")

        wait_for_engine(DEEPSEEK_PORT)
        current_model = model_name
        print(f"[BSAI Manager] Model {model_name} is now active on port {DEEPSEEK_PORT}!")
        return

    # --- Strata 模型：原地重启 8080 上的 Strata 引擎；停掉 DeepSeek(8083) 释放显存；Flint(8082) 不受影响 ---
    with lock:
        # 停掉 8083 上的 DeepSeek（Qwen 与 DeepSeek 同占大显存，不并行）
        if deepseek_process:
            try:
                subprocess.run(["taskkill", "/f", "/t", "/pid", str(deepseek_process.pid)],
                             capture_output=True)
                deepseek_process.wait(timeout=5)
            except:
                try:
                    deepseek_process.kill()
                except:
                    pass
            deepseek_process = None

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
        
        # BSAI: 联网搜索 MCP 自包含注入 —— 不依赖任何外部 shim。
        # 运行卡 json 无 mcp_servers 时，用当前 python 补一个 Bing websearch 工具；
        # 任何电脑装好插件后（tools/ddg_websearch_mcp.py 就位）即可联网搜索。
        try:
            cfg_m = json.loads(json_file.read_text(encoding="utf-8-sig"))
            if not isinstance(cfg_m.get("mcp_servers"), dict) or not cfg_m.get("mcp_servers"):
                ws = INSTALL_DIR / "tools" / "ddg_websearch_mcp.py"
                if ws.is_file():
                    cfg_m["mcp_servers"] = {"websearch": {
                        "command": sys.executable,
                        "args": ["-u", str(ws)],
                        "env": {},
                        "cwd": str(INSTALL_DIR),
                    }}
                    json_file.write_text(
                        json.dumps(cfg_m, indent=4, ensure_ascii=False),
                        encoding="utf-8")
                    print(f"[BSAI Manager] websearch MCP 已注入: {ws.name}")
        except Exception as e:
            print(f"[BSAI Manager] MCP 注入跳过: {e}")
        
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
        # Flint -> 等 8082；DeepSeek -> 等 8083；其余（Strata）-> 等 8080
        if target_model == "flint":
            wait_port = FLINT_PORT
        elif target_model == "deepseek":
            wait_port = DEEPSEEK_PORT
        else:
            wait_port = STRATA_PORT
        if parsed.path == "/" or parsed.path == "/wait":
            # Waiting page: auto-redirect when target engine is ready
            check_url = f"http://127.0.0.1:{wait_port}/v1/models"
            # 破缓存：每次跳转带时间戳，确保浏览器拿到最新的 BSAI 工作台页面
            redirect_url = f"http://127.0.0.1:{wait_port}/?_={int(time.time() * 1000)}"
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
        elif parsed.path == "/api/websearch":
            q = (qs.get("q", [""])[0] or "").strip()
            if not q:
                self.send_json({"error": "缺少 q 参数", "results": []}, 400)
                return
            try:
                self.send_json({"query": q, "results": bing_search(q, 6)})
            except Exception as e:
                print(f"[BSAI Manager] websearch 失败: {e}")
                self.send_json({"query": q, "results": [], "error": str(e)})
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

    restore_ui_patches()

    print(f"[BSAI Manager] Starting with model: {initial_model}")
    print(f"[BSAI Manager] Waiting page: http://127.0.0.1:{MANAGER_PORT}/")
    print(f"[BSAI Manager] Web UI will open automatically when ready")
    
    # Start HTTP manager server first (so waiting page is available)
    # 线程化：联网搜索最长 8 秒，不能把模型切换 / 状态轮询一起堵住
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", MANAGER_PORT), ManagerHandler)
    httpd.daemon_threads = True
    httpd.allow_reuse_address = True
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
