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

                # Step 3.5: inject BSAI logo into Strata web UI
                try:
                    plugin_web = os.path.join(os.path.dirname(__file__), "web", "bsai-logo.png")
                    strata_web = os.path.join(install, "serve", "web", "bsai-logo.png")
                    if os.path.exists(plugin_web) and not os.path.exists(strata_web):
                        import shutil
                        shutil.copy2(plugin_web, strata_web)
                        log.info("[BSAI-LLM-Strata] BSAI logo injected into Strata web UI.")
                except Exception as e:
                    log.debug(f"[BSAI-LLM-Strata] Logo inject skipped: {e}")

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
