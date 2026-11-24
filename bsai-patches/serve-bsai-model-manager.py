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
STRATA_PORT = 8080   # BSAI Strata engine / BSAI web UI (restarted in place when switching models, port kept)
FLINT_PORT = 8082    # Flint 40B standalone llama.cpp engine (runs beside Strata on its own port)
DEEPSEEK_PORT = 8083 # DeepSeek V4.1 standalone strata-dsv41 engine (runs beside Strata on its own port)
INSTALL_DIR = Path(__file__).parent
PYTHON_EXE = sys.executable

# Model options
MODELS = {
    "Q2_0": {"json": "strata-q2_0.json", "desc": "fastest"},
    "IQ2_XS": {"json": "strata-iq2_xs.json", "desc": "better"},
    "IQ3_XXS": {"json": "strata-iq3_xxs.json", "desc": "good"},
    "IQ3_S": {"json": "strata-iq3_s.json", "desc": "best quality"},
    "flint": {"json": None, "desc": "Flint 40B coding model"},
    "deepseek": {"json": "strata-deepseek-v4.1-flash-ist-das-2.00bpw.json",
                 "desc": "DeepSeek V4.1 reasoning"},
}

# Flint 40B: standalone llama.cpp engine (not Strata), fixed port FLINT_PORT=8082, never touches 8080
FLINT_MODEL_CANDIDATES = ("flint-UD-Q3_K_XL.gguf", "flint-UD-Q4_K_XL.gguf")
FLINT_ALIAS = "Flint-40B"


def _integration_root():
    """Portable package root (next to the launcher exe) - resolved relatively so it works across drives."""
    p = Path(__file__).resolve().parents[4]
    return p


def _pick_engine_dir(base):
    """BSAI-llama.cpp subfolder picked for the GPU compute capability / driver (CUDA 13 and CUDA 12.4 builds)."""
    sel = base / "engine-selection.json"
    if not sel.is_file():
        return base
    try:
        import json
        rel = str(json.loads(sel.read_text(encoding="utf-8")).get("selected") or ".")
    except Exception:
        return base
    if rel in (".", "", "None", "none"):
        return base
    cand = base / rel
    return cand if (cand / "llama-server.exe").exists() else base


def _find_llama_server():
    """Look for a portable llama-server.exe under the package root (BSAI-llama.cpp first)."""
    root = _integration_root()
    for rel in ("BSAI-llama.cpp", "llama.cpp"):
        base = root / rel
        if not base.exists():
            continue
        cand = _pick_engine_dir(base) / "llama-server.exe"
        if cand.exists():
            return str(cand)
    raise FileNotFoundError(
        "llama-server.exe not found (package root/BSAI-llama.cpp or /llama.cpp): %s" % root)


def _find_flint_model():
    """Find the Flint GGUF under models/LLM/flint/ (Q3_K_XL preferred)."""
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
        "Flint GGUF not found (%s under %s)" % (base / "LLM" / "flint",
                                     " / ".join(FLINT_MODEL_CANDIDATES)))

# ------------------------------------------------------------------ BSAI web search
# The SEARCH: flow in the web page calls GET /api/websearch?q=...; the backend scrapes
# Bing (reachable from mainland China; DuckDuckGo/Google time out), stdlib only, 8s timeout, empty result on failure.
SEARCH_TIMEOUT = 8
SEARCH_HOSTS = ("cn.bing.com", "www.bing.com")
SEARCH_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


class _BingParser(HTMLParser):
    """Pull title / link / snippet from the <li class="b_algo"> entries on the Bing results page."""

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
    """Bing scrape, no API key; cn.bing.com first, then www.bing.com."""
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
            print(f"[BSAI Manager] websearch {host} failed: {e}")
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
    raise RuntimeError("search failed: %s" % (last or "no results parsed"))


strata_process = None   # Strata engine on 8080
flint_process = None     # Flint llama.cpp engine on 8082 (separate process)
deepseek_process = None  # DeepSeek strata-dsv41 engine on 8083 (separate process)
current_model = None
lock = threading.Lock()


def _pids_listening_on(port):
    """Return the set of PIDs LISTENING on the given TCP port."""
    pids = set()
    try:
        # netstat on Chinese Windows writes GBK/OEM; with PYTHONUTF8=1 text=True decodes as UTF-8
        # and raises UnicodeDecodeError (then stdout becomes None in the reader thread -> AttributeError).
        # So pass encoding + errors="replace" explicitly: OEM code page on Windows, locale elsewhere.
        enc = "oem" if os.name == "nt" else "utf-8"
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True,
                             encoding=enc, errors="replace").stdout
    except Exception:
        try:
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                 capture_output=True, text=True,
                                 errors="replace").stdout
        except Exception:
            return pids
    if not out:
        return pids
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].rsplit(":", 1)[-1] == str(port):
                try:
                    pids.add(int(parts[4]))
                except ValueError:
                    pass
    return pids


def free_port(port, timeout=15):
    """Kill leftover processes holding the port and wait until the port is released.

    A manager restart drops in-memory state such as deepseek_process/flint_process, so old
    instances cannot be found that way; a leftover instance still holding the fixed port makes
    server.py exit with "port already in use". Cleaning by port before launch removes that error.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        pids = _pids_listening_on(port)
        if not pids:
            return True
        for pid in pids:
            try:
                subprocess.run(["taskkill", "/f", "/t", "/pid", str(pid)], capture_output=True)
            except Exception:
                pass
        time.sleep(1)
    return not _pids_listening_on(port)


def restore_ui_patches():
    """After an engine update overwrites the install folder, restore the BSAI workbench UI at start-up.

    Patch sources live in the plugin root bsai-patches/ (same as strata_manager.apply_bsai_patches).
    Only the web UI files are restored; serve/server.py keeps the engine-updated version.
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
            print(f"[BSAI Manager] UI patch restored: {name}")


def start_strata(model_name):
    """Start engine. Strata always takes 8080; Flint takes 8082; DeepSeek takes 8083."""
    global strata_process, flint_process, deepseek_process, current_model

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    # --- Flint 40B: standalone llama.cpp engine on port 8082, leaves Strata on 8080 alone ---
    if model_name == "flint":
        with lock:
            # Only restart the old Flint process; the Strata engine keeps running
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
            # Release the port first: kill a leftover llama-server, otherwise the new instance cannot bind 8082
            free_port(FLINT_PORT)

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

    # --- DeepSeek V4.1: standalone strata-dsv41 engine on port 8083; stops Strata on 8080 to free VRAM ---
    if model_name == "deepseek":
        with lock:
            # Stop Strata on 8080 (DeepSeek and Qwen both need large VRAM, so they do not run together)
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
            # Restart the old DeepSeek process
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

            # Release the port first: kill the previous instance/orphan (possibly not started by this manager), otherwise server.py reports port already in use
            free_port(DEEPSEEK_PORT)

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

    # --- Strata model: restart the Strata engine on 8080 in place; stop DeepSeek (8083) to free VRAM; Flint (8082) unaffected ---
    with lock:
        # Stop DeepSeek on 8083 (Qwen and DeepSeek both need large VRAM, so they do not run together)
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
        
        # BSAI: self-contained web-search MCP injection - no external shim needed.
        # When the run-card json has no mcp_servers, add a Bing websearch tool with the current python;
        # any machine with the plugin installed (tools/ddg_websearch_mcp.py present) can search the web.
        try:
            cfg_m = json.loads(json_file.read_text(encoding="utf-8-sig"))
            if not isinstance(cfg_m.get("mcp_servers"), dict) or not cfg_m.get("mcp_servers"):
                ws = INSTALL_DIR / "tools" / "ddg_websearch_mcp.py"
                if ws.is_file():
                    _py = sys.executable
                    try:
                        _py = os.path.relpath(_py, str(INSTALL_DIR))
                    except Exception:
                        pass
                    cfg_m["mcp_servers"] = {"websearch": {
                        "command": _py,
                        "args": ["-u", os.path.join("tools", ws.name)],
                        "env": {},
                        "cwd": ".",
                    }}
                    json_file.write_text(
                        json.dumps(cfg_m, indent=4, ensure_ascii=False),
                        encoding="utf-8")
                    print(f"[BSAI Manager] websearch MCP injected: {ws.name}")
        except Exception as e:
            print(f"[BSAI Manager] MCP injection skipped: {e}")
        
        # Release the port first: kill leftover/orphan instances, otherwise server.py reports port already in use
        free_port(STRATA_PORT)

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
        # Flint -> wait for 8082; DeepSeek -> wait for 8083; everything else (Strata) -> wait for 8080
        if target_model == "flint":
            wait_port = FLINT_PORT
        elif target_model == "deepseek":
            wait_port = DEEPSEEK_PORT
        else:
            wait_port = STRATA_PORT
        if parsed.path == "/" or parsed.path == "/wait":
            # Waiting page: auto-redirect when target engine is ready
            check_url = f"http://127.0.0.1:{wait_port}/v1/models"
            # Cache buster: every redirect carries a timestamp so the browser always gets the latest BSAI workbench page
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
  <div class="title">BSAI Strata is loading...</div>
  <div class="sub">The first start takes 1-2 minutes; the page redirects automatically</div>
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
                self.send_json({"error": "missing q parameter", "results": []}, 400)
                return
            try:
                self.send_json({"query": q, "results": bing_search(q, 6)})
            except Exception as e:
                print(f"[BSAI Manager] websearch failed: {e}")
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
    # Threaded: a web search can take up to 8 seconds and must not block model switching / status polling
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", MANAGER_PORT), ManagerHandler)
    httpd.daemon_threads = True
    httpd.allow_reuse_address = True
    manager_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    manager_thread.start()
    
    # Open the waiting page immediately (carrying the target model: Flint jumps to 8082, Strata to 8080)
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
