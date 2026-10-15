"""BSAI-LLM-Strata entry point."""
from .strata_manager import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    StrataManager,
    DEFAULT_STRATA_BASE_URL,
    DEFAULT_STRATA_MODEL,
)

WEB_DIRECTORY = "./web"

import logging
log = logging.getLogger(__name__)


def _register_routes():
    try:
        import server as _server
        ps = _server.PromptServer.instance
        if ps is None:
            return
        routes = ps.routes
        from aiohttp import web

        async def health(request):
            h = StrataManager.health()
            install = StrataManager.find_install()
            return web.json_response({
                "ok": h.get("ok"),
                "detail": h,
                "install_dir": install,
                "run_scripts": StrataManager.list_run_scripts(),
                "endpoint": DEFAULT_STRATA_BASE_URL,
            })

        async def start(request):
            script = request.query.get("script")
            return web.json_response(StrataManager.start(run_script=script))

        for method, path, handler in [
            ("GET", "/bsai/strata/health", health),
            ("GET", "/bsai/strata/start", start),
        ]:
            try:
                existing = [r for r in routes
                            if getattr(r, "path", None) == path and getattr(r, "method", None) == method]
                if not existing:
                    routes.route(method, path)(handler)
            except Exception:
                pass
        log.info("[BSAI-LLM-Strata] routes mounted: /bsai/strata/health, /bsai/strata/start")
    except Exception as e:
        log.warning("[BSAI-LLM-Strata] route mount deferred: %s", e)


_register_routes()


def _auto_download_engine_async():
    """Background thread: auto-download Strata engine + create .venv on plugin load.
    Wrapped in try-except so any failure never breaks plugin loading."""
    try:
        import threading
        def _worker():
            try:
                # Step 1: find or download Strata source
                install = StrataManager.find_install()
                if install:
                    log.info(f"[BSAI-LLM-Strata] Strata engine found: {install}")
                else:
                    log.warning("[BSAI-LLM-Strata] Strata engine not found, auto-downloading in background…")
                    install = StrataManager.ensure_strata_installed()
                    if install:
                        log.info(f"[BSAI-LLM-Strata] Strata engine auto-download complete: {install}")
                    else:
                        log.warning("[BSAI-LLM-Strata] Auto-download failed. Please manually clone https://github.com/Niko1221/Strata into custom_nodes/BSAI-LLM-Strata/Strata/")
                        return

                # Step 2: create .venv and install Python deps if missing
                venv_python = StrataManager.ensure_strata_venv()
                if venv_python:
                    log.info(f"[BSAI-LLM-Strata] Strata .venv ready: {venv_python}")
                else:
                    log.warning("[BSAI-LLM-Strata] .venv not ready. Please manually run START-HERE.bat in the Strata folder.")
                    return

                # Step 3: download engine binary and prepare model data if missing
                log.info("[BSAI-LLM-Strata] Checking Strata engine binary …")
                ok = StrataManager.ensure_strata_engine()
                if ok:
                    log.info("[BSAI-LLM-Strata] Strata engine fully installed. Ready to use.")
                else:
                    log.warning("[BSAI-LLM-Strata] Engine setup incomplete. Please manually run START-HERE.bat.")
                    return

                # Step 3.5: inject BSAI logo + patch Strata web UI
                try:
                    import shutil
                    # Copy logo file
                    plugin_web = os.path.join(os.path.dirname(__file__), "web", "bsai-logo.png")
                    strata_web = os.path.join(install, "serve", "web", "bsai-logo.png")
                    if os.path.exists(plugin_web):
                        shutil.copy2(plugin_web, strata_web)

                    # Patch server.py: add .png to supported static file types
                    server_py = os.path.join(install, "serve", "server.py")
                    if os.path.exists(server_py):
                        with open(server_py, "r", encoding="utf-8") as f:
                            content = f.read()
                        # Check if already patched
                        if '".png": "image/png"' not in content:
                            old_types = '".svg": "image/svg+xml"}'
                            new_types = '".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon"}'
                            content = content.replace(old_types, new_types)
                            with open(server_py, "w", encoding="utf-8") as f:
                                f.write(content)
                            log.info("[BSAI-LLM-Strata] Patched server.py to support .png files.")

                    # Patch index.html: BSAI logo + title
                    index_html = os.path.join(install, "serve", "web", "index.html")
                    if os.path.exists(index_html):
                        with open(index_html, "r", encoding="utf-8") as f:
                            html = f.read()
                        # Replace brand section if not already patched
                        if "百声AI Strata" not in html:
                            old_brand = '''<span class="st-header__brand" style="display:flex;align-items:center;gap:8px;">
    <img src="web/bsai-logo.png" alt="BSAI" style="width:28px;height:28px;border-radius:50%;vertical-align:middle;">
    Strata
  </span>'''
                            new_brand = '''<span class="st-header__brand" style="display:flex;align-items:center;gap:8px;">
    <img src="web/bsai-logo.png?v=2" alt="BSAI" style="width:32px;height:32px;border-radius:50%;vertical-align:middle;">
    <strong style="font-size:16px;">百声AI Strata</strong>
  </span>'''
                            html = html.replace(old_brand, new_brand)
                            with open(index_html, "w", encoding="utf-8") as f:
                                f.write(html)
                            log.info("[BSAI-LLM-Strata] Patched index.html with BSAI branding.")

                except Exception as e:
                    log.debug(f"[BSAI-LLM-Strata] Web UI patch skipped: {e}")

                # Step 4: auto-start Strata service in background
                import socket as _socket
                s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                s.settimeout(2)
                result = s.connect_ex(("127.0.0.1", 8080))
                s.close()
                if result != 0:
                    log.info("[BSAI-LLM-Strata] Strata service not running, auto-starting in background…")
                    try:
                        start_result = StrataManager.start()
                        if start_result.get("ok"):
                            log.info(f"[BSAI-LLM-Strata] Strata service started (pid={start_result.get('pid')})")
                        else:
                            log.warning(f"[BSAI-LLM-Strata] Strata auto-start failed: {start_result}")
                    except Exception as e:
                        log.warning(f"[BSAI-LLM-Strata] Strata auto-start error: {e}")
                else:
                    log.info("[BSAI-LLM-Strata] Strata service already running.")
            except Exception as e:
                log.warning(f"[BSAI-LLM-Strata] Auto-download error: {e}")
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
    except Exception as e:
        log.warning(f"[BSAI-LLM-Strata] Failed to start auto-download thread: {e}")


_auto_download_engine_async()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY",
           "StrataManager", "DEFAULT_STRATA_BASE_URL", "DEFAULT_STRATA_MODEL"]
