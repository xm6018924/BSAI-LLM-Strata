"""BSAI-LLM-Strata entry point."""
from .strata_manager import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    StrataManager,
    NemotronManager,
    DEFAULT_STRATA_BASE_URL,
    DEFAULT_STRATA_MODEL,
    NEMOTRON_BASE_URL,
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
            # Only this install's own run-*.bat files can be launched: the quant label
            # picks one of them, and a caller-supplied path is never passed through.
            quant = request.query.get("quant")
            script = StrataManager._bat_for_quant(quant) if quant else None
            return web.json_response(StrataManager.start(run_script=script))

        async def install(request):
            """User-initiated install: pinned upstream commit + hash-checked engine.

            Nothing is downloaded at plugin load or by any GET route; the install only
            starts when this POST is called explicitly.
            """
            import threading

            def _work():
                try:
                    install_dir = StrataManager.ensure_strata_installed()
                    if not install_dir:
                        log.warning("[BSAI-LLM-Strata] engine source install failed")
                        return
                    StrataManager.ensure_strata_venv()
                    if StrataManager.ensure_strata_engine():
                        log.info("[BSAI-LLM-Strata] engine install finished and verified")
                    else:
                        log.warning("[BSAI-LLM-Strata] engine install did not complete; see the log")
                except Exception as exc:
                    log.warning("[BSAI-LLM-Strata] install failed: %s", exc)

            threading.Thread(target=_work, daemon=True).start()
            return web.json_response({"ok": True, "started": True})

        async def nemotron_health(request):
            h = NemotronManager.health()
            return web.json_response({
                "ok": h.get("ok"),
                "detail": h,
                "endpoint": NEMOTRON_BASE_URL,
                "engine": NemotronManager.find_engine(),
                "model": NemotronManager.find_model(),
                "mmproj": NemotronManager.find_mmproj(),
            })

        async def nemotron_start(request):
            ok, msg = NemotronManager.ensure_running()
            return web.json_response({"ok": ok, "message": msg, "endpoint": NEMOTRON_BASE_URL})

        async def nemotron_models(request):
            return web.json_response({"models": NemotronManager.models()})

        async def nemotron_stop(request):
            return web.json_response(NemotronManager.stop())

        for method, path, handler in [
            ("GET", "/bsai/strata/health", health),
            ("GET", "/bsai/strata/start", start),
            ("POST", "/bsai/strata/install", install),
            ("GET", "/bsai/nemotron/health", nemotron_health),
            ("GET", "/bsai/nemotron/start", nemotron_start),
            ("GET", "/bsai/nemotron/models", nemotron_models),
            ("GET", "/bsai/nemotron/stop", nemotron_stop),
        ]:
            try:
                existing = [r for r in routes
                            if getattr(r, "path", None) == path and getattr(r, "method", None) == method]
                if not existing:
                    routes.route(method, path)(handler)
            except Exception:
                pass
        log.info("[BSAI-LLM-Strata] routes mounted: /bsai/strata/*, /bsai/nemotron/*")
    except Exception as e:
        log.warning("[BSAI-LLM-Strata] route mount deferred: %s", e)


_register_routes()


def _generate_strata_launcher():
    """Regenerate the BSAI_Strata service launcher at ComfyUI start-up.

    The launcher sits next to the launcher exe of the portable package and uses only
    relative paths, so a double click starts the service manually. The plugin never
    downloads or starts anything at load time.
    """
    try:
        StrataManager.ensure_bsai_launcher()
    except Exception as e:
        log.warning(f"[BSAI-LLM-Strata] launcher generation skipped: {e}")


_generate_strata_launcher()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY",
           "StrataManager", "NemotronManager", "DEFAULT_STRATA_BASE_URL",
           "DEFAULT_STRATA_MODEL", "NEMOTRON_BASE_URL"]
