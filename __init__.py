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
            script = request.query.get("script")
            return web.json_response(StrataManager.start(run_script=script))

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

                    # Patch requirements.txt: relax pinned versions (2026 versions may not exist on mirrors)
                    req_file = os.path.join(install, "requirements.txt")
                    if os.path.exists(req_file):
                        with open(req_file, "r", encoding="utf-8") as f:
                            req_content = f.read()
                        # Replace all == with >= to relax version constraints
                        import re
                        req_content_new = re.sub(r'==([0-9])', r'>=\1', req_content)
                        if req_content_new != req_content:
                            with open(req_file, "w", encoding="utf-8") as f:
                                f.write(req_content_new)
                            log.info("[BSAI-LLM-Strata] Patched requirements.txt: relaxed all version pins.")

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

                    # Patch app.js: support video uploads
                    app_js = os.path.join(install, "serve", "web", "app.js")
                    if os.path.exists(app_js):
                        with open(app_js, "r", encoding="utf-8") as f:
                            js = f.read()
                        if "/v1/video/upload" not in js:
                            old_addfiles = '''function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {'''
                            new_addfiles = '''function addFiles(files) {
  for (const f of files) {
    // Video files: upload to backend for frame extraction
    if (f.type.startsWith("video/") || /\\.(mp4|webm|mov|avi|mkv)$/i.test(f.name)) {
      if (f.size > 100 * 1024 * 1024) { toast("warn", "Video too large", `${f.name} is over 100 MB.`); continue; }
      toast("info", "Video uploading", `${f.name}: uploading for frame extraction…`);
      const fd = new FormData();
      fd.append("file", f);
      fetch("/v1/video/upload", { method: "POST", body: fd })
        .then((r) => r.json())
        .then((data) => {
          if (data.ok && data.frames && data.frames.length) {
            toast("success", "Video processed", `${f.name}: extracted ${data.frames.length} frames.`);
            data.frames.forEach((frame) => {
              attachments.push({kind: "image", name: `Frame (${frame.timestamp}s)`, url: frame.url});
            });
            renderAttachments();
          } else {
            toast("warn", "Video failed", data.error || "Could not process video.");
          }
        })
        .catch((e) => toast("warn", "Video error", String(e)));
      continue;
    }
    if (f.type.startsWith("image/")) {'''
                            js = js.replace(old_addfiles, new_addfiles)
                            with open(app_js, "w", encoding="utf-8") as f:
                                f.write(js)
                            log.info("[BSAI-LLM-Strata] Patched app.js with video upload support.")

                    # Create Strata launcher bat in ComfyUI root dir
                    try:
                        import folder_paths
                        comfy_root = os.path.dirname(folder_paths.base_path) if hasattr(folder_paths, 'base_path') else os.path.dirname(os.path.dirname(install))
                        bat_src = os.path.join(os.path.dirname(__file__), "web", "启动BSAI_Strata服务.bat")
                        bat_dst = os.path.join(comfy_root, "启动BSAI_Strata服务.bat")
                        if os.path.exists(bat_src) and not os.path.exists(bat_dst):
                            import shutil
                            shutil.copy2(bat_src, bat_dst)
                            log.info(f"[BSAI-LLM-Strata] Created Strata launcher: {bat_dst}")
                    except Exception as e:
                        log.debug(f"[BSAI-LLM-Strata] Launcher bat creation skipped: {e}")

                except Exception as e:
                    log.debug(f"[BSAI-LLM-Strata] Web UI patch skipped: {e}")

                # Step 4: 不在 ComfyUI 启动时拉起 Strata（按需启动）。
                # 加载 125B MoE 需要 1-3 分钟且占用大量显存/内存，开机自启会拖慢启动。
                # 引擎改为首次被工作流用到时由节点执行路径自动拉起：
                #   - BSAI-LLM-Strata 节点 → StrataManager.ensure_running()
                #   - BSAI-FLUX3-ComfyUI 增强器 → ensure_strata()
                #   - 也可手动运行「启动BSAI_Strata服务.bat」或前端按钮
                log.info("[BSAI-LLM-Strata] Install check done. Strata engine will start on demand (first workflow use).")
            except Exception as e:
                log.warning(f"[BSAI-LLM-Strata] Auto-download error: {e}")
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
    except Exception as e:
        log.warning(f"[BSAI-LLM-Strata] Failed to start auto-download thread: {e}")


_auto_download_engine_async()


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY",
           "StrataManager", "NemotronManager", "DEFAULT_STRATA_BASE_URL",
           "DEFAULT_STRATA_MODEL", "NEMOTRON_BASE_URL"]
