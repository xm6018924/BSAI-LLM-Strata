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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY",
           "StrataManager", "DEFAULT_STRATA_BASE_URL", "DEFAULT_STRATA_MODEL"]
