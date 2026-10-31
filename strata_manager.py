"""BSAI-LLM-Strata: local Strata (Qwen3.8-Flash-Next 125B MoE) integration for ComfyUI.

Exposes:
  - StrataManager: process lifecycle + health probe (auto-detect running Strata at 127.0.0.1:8080)
  - BSAI_LLM_Call: chat completion node
  - BSAI_LLM_VisionDescribe: image -> text (uses mmproj)
  - BSAI_LLM_PromptEnhance: prompt enhancement
  - Web routes: /bsai/strata/health, /bsai/strata/config
"""
import os
import sys
import json
import time
import shutil
import socket as _socket
import http.client as _http_client
import subprocess
import urllib.request
import urllib.error
from urllib.parse import urlparse as _urlparse
import threading

import folder_paths


# ---------------------------------------------------------------------------
# Blocking-socket HTTP helper (avoids WinError 10035 under Python 3.14 asyncio)
# ---------------------------------------------------------------------------
def _blocking_get_json(url, timeout=5.0, headers=None):
    """GET a JSON URL using an explicit blocking socket (bypasses urllib)."""
    p = _urlparse(url)
    host, port = p.hostname, p.port or 80
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.setblocking(True)
    s.connect((host, port))
    try:
        conn = _http_client.HTTPConnection(host, port, timeout=timeout)
        conn.sock = s
        h = {"Accept": "application/json"}
        if headers:
            h.update(headers)
        conn.request("GET", p.path or "/", headers=h)
        r = conn.getresponse()
        body = r.read().decode("utf-8", "replace")
        return r.status, body
    finally:
        try:
            s.close()
        except Exception:
            pass


def _blocking_post_json(url, body_bytes, timeout=60.0, headers=None):
    """POST JSON using an explicit blocking socket."""
    p = _urlparse(url)
    host, port = p.hostname, p.port or 80
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.setblocking(True)
    s.connect((host, port))
    try:
        conn = _http_client.HTTPConnection(host, port, timeout=timeout)
        conn.sock = s
        h = {"Content-Type": "application/json", "Content-Length": str(len(body_bytes))}
        if headers:
            h.update(headers)
        conn.request("POST", p.path or "/", body=body_bytes, headers=h)
        r = conn.getresponse()
        body = r.read().decode("utf-8", "replace")
        return r.status, body
    finally:
        try:
            s.close()
        except Exception:
            pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_STRATA_HOST = "127.0.0.1"
DEFAULT_STRATA_PORT = 8080
DEFAULT_STRATA_BASE_URL = f"http://{DEFAULT_STRATA_HOST}:{DEFAULT_STRATA_PORT}/v1"
DEFAULT_STRATA_MODEL = "qwen3.8-flash-next"

# Where the user may have cloned https://github.com/Niko1221/Strata
# Priority 0: the bundled copy shipped with this plugin (one-stop install).
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
_STRATA_SEARCH_PATHS = [
    os.path.join(_PLUGIN_DIR, "Strata"),  # bundled runtime
    r"G:\Strata",
    r"C:\Strata",
    r"D:\Strata",
    r"E:\Strata",
    r"F:\Strata",
    r"G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\Strata",
    r"G:\BSAI\Strata",
    os.path.expanduser(r"~\Strata"),
    os.path.expanduser(r"~\ProgramMine\Strata"),
]

# Where the model GGUFs live (already on disk per user)
STRATA_MODEL_DIR = os.path.join(folder_paths.models_dir, "LLM", "Qwen3.8-Flash-Next")
# Pre-converted Strata runtime data (packs, tokenizer, MTP) — lives in model dir
# 路径：ComfyUI/models/LLM/Qwen3.8-Flash-Next/Strata-data/
STRATA_DATA_DIR = os.path.join(STRATA_MODEL_DIR, "Strata-data")

# Map quant names → run-*.bat filenames (must match files on disk)
QUANT_OPTIONS = {
    "Auto (自动适配)":             None,  # resolved by detect_best_quant()
    "Q2_0 (最快 / fastest)":      "run-q2_0.bat",
    "IQ2_XS (略好 / better)":     "run-iq2_xs.bat",
    "IQ3_XXS (更好 / good)":      "run-iq3_xxs.bat",
    "IQ3_S (最好 / best quality)": "run-iq3_s.bat",
}
DEFAULT_QUANT_LABEL = "Auto (自动适配)"

# Hardware tiers for auto-select: (min_vram_gb, min_ram_gb, label)
_QUANT_TIERS = [
    (32, 64, "IQ3_S (最好 / best quality)"),
    (24, 48, "IQ3_XXS (更好 / good)"),
    (16, 40, "IQ2_XS (略好 / better)"),
    (0,  0,  "Q2_0 (最快 / fastest)"),  # always fits
]

# Global preferred-quant state file
_PREF_QUANT_FILE = os.path.join(_PLUGIN_DIR, ".preferred_quant.txt")


# ---------------------------------------------------------------------------
# StrataManager
# ---------------------------------------------------------------------------
class StrataManager:
    """Lightweight manager: probe health, optionally spawn Strata if a
    run-<quant>.bat script is found. We do NOT auto-spawn by default because
    loading a 125B MoE takes 1-3 minutes and locks VRAM/RAM."""

    _lock = threading.Lock()
    _proc = None
    _detected_install = None

    # ---------- quant preference ----------
    @staticmethod
    def detect_best_quant():
        """Auto-detect the best fitting quant based on VRAM and RAM.

        Returns a quant label string from QUANT_OPTIONS.
        """
        vram_gb = 0.0
        ram_gb = 0.0
        # VRAM via nvidia-smi
        try:
            import subprocess as _sp
            out = _sp.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
            if lines:
                vram_gb = float(lines[0]) / 1024.0  # MiB → GiB
        except Exception:
            pass
        # RAM via psutil (falls back to os if psutil missing)
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            try:
                import ctypes
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullExtendedVirtual", ctypes.c_ulonglong),
                    ]
                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                ram_gb = stat.ullTotalPhys / (1024 ** 3)
            except Exception:
                ram_gb = 32.0  # conservative default

        for min_vram, min_ram, label in _QUANT_TIERS:
            if vram_gb >= min_vram and ram_gb >= min_ram:
                print(f"[BSAI-LLM-Strata] Auto-select: VRAM={vram_gb:.1f}GB RAM={ram_gb:.1f}GB → {label}")
                return label
        return "Q2_0 (最快 / fastest)"

    @classmethod
    def get_preferred_quant(cls):
        """Read the user's preferred quant label (defaults to Auto)."""
        try:
            with open(_PREF_QUANT_FILE, "r", encoding="utf-8") as f:
                label = f.read().strip()
            if label in QUANT_OPTIONS:
                return label
        except Exception:
            pass
        return DEFAULT_QUANT_LABEL

    @classmethod
    def set_preferred_quant(cls, label):
        """Persist the preferred quant label."""
        if label not in QUANT_OPTIONS:
            return False
        try:
            with open(_PREF_QUANT_FILE, "w", encoding="utf-8") as f:
                f.write(label)
            return True
        except Exception:
            return False

    @classmethod
    def _bat_for_quant(cls, label):
        """Resolve a quant label to a full run-*.bat path. 'Auto' is resolved."""
        if label == "Auto (自动适配)":
            label = cls.detect_best_quant()
        if label not in QUANT_OPTIONS:
            return None
        install = cls.find_install()
        if not install:
            return None
        bat_name = QUANT_OPTIONS.get(label)
        if not bat_name:
            return None
        path = os.path.join(install, bat_name)
        return path if os.path.exists(path) else None

    # ---------- discovery ----------
    @classmethod
    def ensure_strata_installed(cls):
        """Auto-clone Strata engine into the plugin's Strata/ subfolder if missing.
        Returns the install path, or None on failure."""
        target = os.path.join(_PLUGIN_DIR, "Strata")
        # Already there?
        if os.path.isdir(target) and os.path.exists(os.path.join(target, "setup.py")):
            # Apply BSAI custom patches (video analysis, UI improvements, etc.)
            cls.apply_bsai_patches(target)
            return target
        # Clone it - try GitHub first, then mirrors
        mirrors = [
            "https://github.com/Niko1221/Strata.git",
            "https://ghproxy.com/https://github.com/Niko1221/Strata.git",
            "https://mirror.ghproxy.com/https://github.com/Niko1221/Strata.git",
            "https://gh-proxy.com/https://github.com/Niko1221/Strata.git",
        ]
        for url in mirrors:
            try:
                os.makedirs(target, exist_ok=True)
                print(f"[BSAI-LLM-Strata] 首次使用，正在自动下载 Strata 引擎 …")
                print(f"[BSAI-LLM-Strata] 尝试源: {url}")
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, target],
                    check=True, timeout=300,
                )
                print("[BSAI-LLM-Strata] Strata 引擎下载完成。")
                # Apply BSAI custom patches after fresh clone
                cls.apply_bsai_patches(target)
                return target
            except Exception as e:
                print(f"[BSAI-LLM-Strata] 下载失败: {e}")
                # Clean up failed clone
                import shutil
                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                continue
        print("[BSAI-LLM-Strata] 所有下载源都失败了，请检查网络连接。")
        return None

    @staticmethod
    def apply_bsai_patches(strata_dir):
        """Apply BSAI custom patches to Strata submodule (video analysis, UI improvements, etc.)"""
        patches_dir = os.path.join(_PLUGIN_DIR, "bsai-patches")
        if not os.path.isdir(patches_dir):
            return
        # Patch file mapping: (source_patch_file, target_relative_path)
        patches = [
            ("serve-web-app.js", os.path.join("serve", "web", "app.js")),
            ("serve-web-index.html", os.path.join("serve", "web", "index.html")),
            ("serve-web-bsai-logo.png", os.path.join("serve", "web", "bsai-logo.png")),
            ("serve-server.py", os.path.join("serve", "server.py")),
            ("serve-bsai-model-manager.py", "bsai_model_manager.py"),
        ]
        applied = 0
        for patch_name, target_rel in patches:
            patch_path = os.path.join(patches_dir, patch_name)
            target_path = os.path.join(strata_dir, target_rel)
            if not os.path.exists(patch_path):
                continue
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(patch_path, target_path)
                applied += 1
            except Exception as e:
                print(f"[BSAI-LLM-Strata] 应用补丁 {patch_name} 失败: {e}")
        if applied > 0:
            print(f"[BSAI-LLM-Strata] 已应用 {applied} 个 BSAI 自定义补丁。")

    @staticmethod
    def _find_system_python():
        """Find a system Python 3.10+ that supports venv creation.
        ComfyUI embedded Python may not have venv.__main__."""
        candidates = []
        # py launcher
        try:
            r = subprocess.run(["py", "-3", "-c", "import sys; print(sys.executable)"],
                             capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                candidates.append(r.stdout.strip())
        except Exception:
            pass
        # python on PATH
        try:
            r = subprocess.run(["python", "-c", "import sys; print(sys.executable)"],
                             capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                candidates.append(r.stdout.strip())
        except Exception:
            pass
        # Common install paths
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        for ver in ("313", "312", "311", "310"):
            p = os.path.join(local_appdata, "Programs", "Python", f"Python{ver}", "python.exe")
            if os.path.exists(p):
                candidates.append(p)
        # Validate each candidate can run venv
        for p in candidates:
            try:
                r = subprocess.run([p, "-c", "import venv; print('venv OK')"],
                                 capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    return p
            except Exception:
                continue
        return None

    @classmethod
    def ensure_strata_venv(cls):
        """Find system Python and auto-install required dependencies. No venv needed."""
        install = cls.find_install()
        if not install:
            return None
        # Find system Python
        sys_python = cls._find_system_python()
        if not sys_python:
            print("[BSAI-LLM-Strata] 找不到系统 Python 3.10+。")
            print("[BSAI-LLM-Strata] 请手动安装 Python 3.10+ 后重启 ComfyUI。")
            return None
        # Verify deps are installed
        try:
            r = subprocess.run(
                [sys_python, "-c", "import numpy, jinja2, yaml, requests; print('deps OK')"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"[BSAI-LLM-Strata] 使用系统 Python: {sys_python}")
                return sys_python
            else:
                # Auto-install missing dependencies using Tsinghua mirror
                print("[BSAI-LLM-Strata] 检测到缺少依赖，正在自动安装 ...")
                deps = "numpy jinja2 pyyaml tqdm requests cmake ninja pillow psutil markupsafe certifi charset-normalizer idna urllib3 colorama regex"
                env = os.environ.copy()
                env["PIP_INDEX_URL"] = "https://pypi.tuna.tsinghua.edu.cn/simple"
                env["PIP_TRUSTED_HOST"] = "pypi.tuna.tsinghua.edu.cn"
                r2 = subprocess.run(
                    [sys_python, "-m", "pip", "install", deps],
                    capture_output=True, text=True, timeout=300,
                    env=env,
                )
                if r2.returncode == 0:
                    print("[BSAI-LLM-Strata] 依赖安装完成。")
                    return sys_python
                else:
                    print(f"[BSAI-LLM-Strata] 依赖安装失败: {r2.stderr[-300:]}")
                    return None
        except Exception as e:
            print(f"[BSAI-LLM-Strata] 检查依赖失败: {e}")
            return None

    @staticmethod
    def _detect_available_quants():
        """Scan model directory and find which quants are already downloaded.
        Returns list of (quant_name, gguf_dir) tuples, sorted by quality ascending."""
        if not os.path.isdir(STRATA_MODEL_DIR):
            return []
        found = []
        # Check root directory and all subdirectories
        search_dirs = [STRATA_MODEL_DIR]
        for d in os.listdir(STRATA_MODEL_DIR):
            sub = os.path.join(STRATA_MODEL_DIR, d)
            if os.path.isdir(sub):
                search_dirs.append(sub)
        # Scan each directory for GGUF files
        for d in search_dirs:
            try:
                files = os.listdir(d)
            except Exception:
                continue
            # Look for quant patterns in filenames
            for f in files:
                low = f.lower()
                if not low.endswith(".gguf"):
                    continue
                # Detect quant type
                if "iq3_s" in low or "iq3-s" in low:
                    quant = "IQ3_S"
                elif "iq3_xxs" in low or "iq3-xxs" in low:
                    quant = "IQ3_XXS"
                elif "iq2_xs" in low or "iq2-xs" in low:
                    quant = "IQ2_XS"
                elif "q2_0" in low or "q2-0" in low:
                    quant = "Q2_0"
                elif "q4_xs" in low or "q4-xs" in low:
                    quant = "Q4_XS"
                else:
                    continue
                # Make sure it's a model GGUF (not mmproj, not mtp)
                if "mmproj" in low or "mtp" in low:
                    continue
                # Only add once per quant
                if not any(q == quant for q, _ in found):
                    found.append((quant, d))
                break  # one file per dir is enough to confirm
        # Sort by quality: IQ3_S highest quality first, Q2_0 fastest last
        order = {"IQ3_S": 0, "IQ3_XXS": 1, "IQ2_XS": 2, "Q2_0": 3, "Q4_XS": 4}
        found.sort(key=lambda x: order.get(x[0], 99))
        return found

    @classmethod
    def ensure_strata_engine(cls):
        """Auto-run setup.py to download engine binary and prepare model data.
        Auto-detects available quants and uses the best one.
        Auto-retries once on failure.
        Returns True on success, False on failure."""
        install = cls.find_install()
        if not install:
            return False
        # Use system Python directly, no venv
        venv_python = cls.ensure_strata_venv()
        if not venv_python:
            return False
        # First: check if engine files exist in model directory's engine/ folder
        # If yes, copy them to plugin's Strata/engine/
        # Engine files can be directly in model dir's engine/ folder (simple layout)
        # OR in Strata-data/engine/ (legacy layout)
        model_engine_paths = [
            os.path.join(STRATA_MODEL_DIR, "engine"),
            os.path.join(STRATA_DATA_DIR, "engine"),
        ]
        plugin_engine_dir = os.path.join(install, "engine")
        for model_engine_dir in model_engine_paths:
            if os.path.isdir(model_engine_dir):
                model_exe = os.path.join(model_engine_dir, "strata.exe")
                if os.path.exists(model_exe):
                    print(f"[BSAI-LLM-Strata] 检测到模型目录下的引擎文件: {model_engine_dir}")
                    # Create plugin engine dir if not exists
                    os.makedirs(plugin_engine_dir, exist_ok=True)
                    # Copy all files from model engine dir to plugin engine dir
                    import shutil
                    for f in os.listdir(model_engine_dir):
                        src = os.path.join(model_engine_dir, f)
                        dst = os.path.join(plugin_engine_dir, f)
                        if os.path.isfile(src):
                            shutil.copy2(src, dst)
                    print(f"[BSAI-LLM-Strata] 引擎文件已复制到插件目录: {plugin_engine_dir}")
                    break

        # Check if run-*.bat already exists (engine already set up)
        existing_runs = [f for f in os.listdir(install) if f.startswith("run-") and f.endswith(".bat")]
        if existing_runs:
            return True
        # Check if engine binary already exists (e.g. copied manually)
        # Check both common locations: engine/strata.exe and ./strata.exe
        strata_exe_paths = [
            os.path.join(install, "engine", "strata.exe"),
            os.path.join(install, "strata.exe"),
        ]
        strata_exe = None
        for p in strata_exe_paths:
            if os.path.exists(p):
                strata_exe = p
                break
        if strata_exe:
            print("[BSAI-LLM-Strata] 检测到引擎二进制已存在，跳过下载步骤 …")
            # Auto-detect available quants and generate json config + run-*.bat
            available = cls._detect_available_quants()
            if available:
                best_quant, best_dir = available[0]
                print(f"[BSAI-LLM-Strata] 检测到已有的量化模型: {best_quant}")
                # Generate json config
                ok_gen, msg_gen = cls._refresh_json_config(best_quant)
                if ok_gen:
                    print(f"[BSAI-LLM-Strata] 配置文件生成完成。")
                    # Generate run-*.bat
                    bat_name = f"run-{best_quant.lower().replace('_', '_')}.bat"
                    # Map quant name to bat filename
                    quant_map = {"Q2_0": "run-q2_0.bat", "IQ2_XS": "run-iq2_xs.bat",
                                 "IQ3_XXS": "run-iq3_xxs.bat", "IQ3_S": "run-iq3_s.bat"}
                    bat_name = quant_map.get(best_quant, f"run-{best_quant.lower()}.bat")
                    bat_path = os.path.join(install, bat_name)
                    if not os.path.exists(bat_path):
                        bat_content = f'''@echo off
title BSAI Strata {best_quant}
cd /d "{install}"
"{venv_python}" "bsai_model_manager.py" "{best_quant}"
if errorlevel 1 pause
'''
                        with open(bat_path, "w", encoding="utf-8") as f:
                            f.write(bat_content)
                        print(f"[BSAI-LLM-Strata] 启动脚本已生成: {bat_name}")
                    return True
                else:
                    print(f"[BSAI-LLM-Strata] 配置文件生成失败: {msg_gen}")
            else:
                print("[BSAI-LLM-Strata] 未检测到已有量化模型，需要运行 setup.py 准备模型数据。")
        # Auto-detect available quants
        available = cls._detect_available_quants()
        if available:
            # Pick the fastest available quant (first in sorted list)
            best_quant, best_dir = available[0]
            print(f"[BSAI-LLM-Strata] 检测到已有的量化模型: {best_quant}")
            print(f"[BSAI-LLM-Strata] 模型目录: {best_dir}")
        else:
            # No pre-downloaded models, let setup.py download default
            best_quant = None
            best_dir = None
            print("[BSAI-LLM-Strata] 未检测到已有模型，setup.py 将自动下载默认模型 …")
        # Auto-install all dependencies using Tsinghua mirror before running setup.py
        print("[BSAI-LLM-Strata] 正在自动安装 Strata 依赖包 ...")
        deps_list = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "pillow", "psutil", "markupsafe", "certifi", "charset-normalizer", "idna", "urllib3", "colorama"]
        env = os.environ.copy()
        env["PIP_INDEX_URL"] = "https://pypi.tuna.tsinghua.edu.cn/simple"
        env["PIP_TRUSTED_HOST"] = "pypi.tuna.tsinghua.edu.cn"
        r_deps = subprocess.run(
            [venv_python, "-m", "pip", "install"] + deps_list,
            capture_output=True, text=True, timeout=300,
            env=env,
        )
        print(f"[BSAI-LLM-Strata] pip install stdout: {r_deps.stdout[-500:]}")
        print(f"[BSAI-LLM-Strata] pip install stderr: {r_deps.stderr[-500:]}")
        if r_deps.returncode == 0:
            print("[BSAI-LLM-Strata] 依赖包安装完成。")
        else:
            print(f"[BSAI-LLM-Strata] 依赖包安装失败，继续尝试运行 setup.py ...")
        # Build setup.py args
        args = [venv_python, "setup.py", "--yes", "--no-start"]
        if best_quant:
            args.extend(["--model", best_quant])
        if best_dir:
            args.extend(["--gguf-dir", best_dir])
        args.extend(["--models-dir", STRATA_MODEL_DIR])
        # Run setup.py with auto-retry
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                if attempt == 1:
                    print("[BSAI-LLM-Strata] 首次使用，正在自动下载 Strata 引擎二进制并准备模型数据 …")
                    print("[BSAI-LLM-Strata] 这可能需要几分钟到十几分钟，请耐心等待 …")
                else:
                    print(f"[BSAI-LLM-Strata] 第 {attempt} 次重试自动安装引擎 …")
                    # Clean up corrupted download files before retrying
                    for f in ["strata-windows-x64.zip", "strata-windows-x64.zip.part"]:
                        p = os.path.join(install, f)
                        if os.path.exists(p):
                            print(f"[BSAI-LLM-Strata] 清理损坏的下载文件: {f}")
                            try:
                                os.remove(p)
                            except Exception:
                                pass
                # Set pip mirror to Tsinghua for automatic dependency installation
                env = os.environ.copy()
                env["PIP_INDEX_URL"] = "https://pypi.tuna.tsinghua.edu.cn/simple"
                env["PIP_TRUSTED_HOST"] = "pypi.tuna.tsinghua.edu.cn"
                # Set GitHub mirror for Strata engine download
                # Try ghproxy.com first, fallback to official GitHub
                if attempt == 1:
                    # First try: official GitHub
                    env["STRATA_PREBUILT_URL"] = "https://github.com/Niko1221/Strata/releases/latest/download/"
                elif attempt == 2:
                    # Second try: ghproxy.com mirror
                    env["STRATA_PREBUILT_URL"] = "https://ghproxy.com/https://github.com/Niko1221/Strata/releases/latest/download/"
                    print("[BSAI-LLM-Strata] 切换到 GitHub 镜像源 (ghproxy.com) ...")
                else:
                    # Third try: mirror.ghproxy.com
                    env["STRATA_PREBUILT_URL"] = "https://mirror.ghproxy.com/https://github.com/Niko1221/Strata/releases/latest/download/"
                    print("[BSAI-LLM-Strata] 切换到 GitHub 镜像源 (mirror.ghproxy.com) ...")
                r = subprocess.run(
                    args,
                    cwd=install,
                    capture_output=True, text=True, timeout=1800,  # 30 min timeout
                    env=env,
                )
                if r.returncode != 0:
                    print(f"[BSAI-LLM-Strata] setup.py 失败 (exit {r.returncode}):")
                    print(f"[BSAI-LLM-Strata] stdout: {r.stdout[-500:] if r.stdout else '(empty)'}")
                    print(f"[BSAI-LLM-Strata] stderr: {r.stderr[-500:] if r.stderr else '(empty)'}")
                    raise subprocess.CalledProcessError(r.returncode, args, output=r.stdout, stderr=r.stderr)
                # Verify run-*.bat was created
                new_runs = [f for f in os.listdir(install) if f.startswith("run-") and f.endswith(".bat")]
                if new_runs:
                    print("[BSAI-LLM-Strata] Strata 引擎安装完成。")
                    return True
                else:
                    print("[BSAI-LLM-Strata] setup.py 运行完成，但未找到 run-*.bat，可能未完全成功。")
                    return False
            except Exception as e:
                print(f"[BSAI-LLM-Strata] 自动安装引擎失败 (尝试 {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    print("[BSAI-LLM-Strata] 等待 10 秒后自动重试 …")
                    import time
                    time.sleep(10)
        print("[BSAI-LLM-Strata] 自动安装引擎失败，请手动运行 START-HERE.bat 完成安装。")
        return False

    @classmethod
    def find_install(cls):
        if cls._detected_install is not None:
            return cls._detected_install
        for p in _STRATA_SEARCH_PATHS:
            if os.path.isdir(p) and (
                os.path.exists(os.path.join(p, "START-HERE.bat"))
                or os.path.exists(os.path.join(p, "setup.py"))
            ):
                cls._detected_install = p
                return p
        # Not found anywhere — auto-clone into plugin dir
        auto = cls.ensure_strata_installed()
        if auto:
            cls._detected_install = auto
            return auto
        return None

    @classmethod
    def list_run_scripts(cls):
        install = cls.find_install()
        if not install:
            return []
        out = []
        for f in os.listdir(install):
            low = f.lower()
            if low.startswith("run-") and low.endswith(".bat"):
                out.append(os.path.join(install, f))
        return sorted(out)

    # ---------- health ----------
    @classmethod
    def health(cls, timeout=1.5):
        url = f"http://{DEFAULT_STRATA_HOST}:{DEFAULT_STRATA_PORT}/v1/models"
        try:
            status, body = _blocking_get_json(url, timeout=timeout)
            data = json.loads(body)
            return {"ok": True, "models": [m.get("id", "?") for m in data.get("data", [])]}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---------- BSAI_Strata服务.bat launcher (regenerated every ComfyUI start) ----------
    _LAUNCHER_BAT_NAME = "BSAI_Strata服务.bat"

    # 不同整合包的 Python 相对候选（按优先级）
    _PY_CANDIDATES_REL = (
        r"python_embeded_cuda\python.exe",   # BSAI Pro 新整合包（G 盘）
        r"python\python.exe",                # 绘世/BSAI 旧整合包（C 盘）
        r"python_embeded\python.exe",        # 标准嵌入式整合包
    )

    @classmethod
    def _package_root(cls):
        """ComfyUI 整合包根目录（绘世启动器.exe 同级）。

        custom_nodes 的上级是 ComfyUI，再上一级即整合包根。
        """
        try:
            cand = os.path.dirname(os.path.dirname(_CUSTOM_NODES_ROOT))
            if os.path.isdir(cand) and (
                os.path.isfile(os.path.join(cand, "绘世启动器.exe"))
                or os.path.isdir(os.path.join(cand, "ComfyUI", "custom_nodes"))
            ):
                return cand
        except Exception:
            pass
        return None

    @classmethod
    def _find_python_rel(cls, root):
        """在整合包根内探测可用的 python.exe，返回相对 root 的路径；找不到返回 None。

        兼容不同环境包：python_embeded_cuda / python / python_embeded / python_envs 多环境目录。
        """
        for rel in cls._PY_CANDIDATES_REL:
            if os.path.isfile(os.path.join(root, rel)):
                return rel
        envs = os.path.join(root, "python_envs")
        if os.path.isdir(envs):
            try:
                for d in sorted(os.listdir(envs)):
                    p = os.path.join(envs, d, "Scripts", "python.exe")
                    if os.path.isfile(p):
                        return os.path.relpath(p, root)
            except Exception:
                pass
        return None

    @classmethod
    def _json_native_ok(cls, install, low):
        """校验 strata-<low>.json 的 --native 主模型文件真实存在（跨机器一致性，
        与 manager available_models 对齐）。"""
        jp = os.path.join(install, f"strata-{low}.json")
        if not os.path.isfile(jp):
            return False
        try:
            with open(jp, "r", encoding="utf-8-sig") as f:
                cfg = json.load(f)
            args = cfg.get("args") or []
            for i, a in enumerate(args):
                if a == "--native" and i + 1 < len(args):
                    return os.path.isfile(args[i + 1])
            return True
        except Exception:
            return False

    @classmethod
    def _pick_quant(cls, install):
        """选择本机实际可用的 quant：优先 preferred/detect，但 json 与
        --native 主模型文件都必须真实存在；否则回退到验证通过的 strata-*.json。

        Returns (quant_upper, quant_lower) e.g. ("IQ3_S", "iq3_s").
        """
        label = cls.get_preferred_quant()
        if label == "Auto (自动适配)":
            label = cls.detect_best_quant()
        pair = cls._QUANT_JSON_MAP.get(label)
        if not pair:
            pair = ("Q2_0", "Q2_0")
        lower, upper = pair[0], pair[1]
        if cls._json_native_ok(install, lower):
            return upper, lower
        # 现有可用 json 兜底（json + 模型文件都真实存在才算可用）
        try:
            jsons = sorted(f for f in os.listdir(install) if f.startswith("strata-") and f.endswith(".json"))
            for jn in jsons:
                low = jn[len("strata-"):-len(".json")]
                if not cls._json_native_ok(install, low):
                    continue
                for (l2, u2) in cls._QUANT_JSON_MAP.values():
                    if l2 == low:
                        return u2, l2
        except Exception:
            pass
        return upper, lower

    @classmethod
    def ensure_bsai_launcher(cls):
        """每次 ComfyUI 启动时调用：在整合包根（绘世启动器.exe 旁）覆盖生成
        BSAI_Strata服务.bat，内容全部为相对路径（基于 %~dp0），Python 与 quant
        按本机环境包自动探测。

        该启动器只等待用户双击运行；插件节点不再自动拉起 Strata。
        """
        try:
            root = cls._package_root()
            if not root:
                print("[BSAI-LLM-Strata] 未定位整合包根目录，跳过启动器生成。")
                return None
            install = cls.find_install()
            if not install:
                print("[BSAI-LLM-Strata] Strata 运行时未安装，跳过启动器生成。")
                return None
            py_rel = cls._find_python_rel(root)
            if not py_rel:
                # 整合包内无 Python：回退到 PATH 中的 python 命令（非相对，极端情况）
                py_rel = "python"
                print("[BSAI-LLM-Strata] 整合包内未找到 python.exe，启动器将使用 PATH 中的 python 命令。")
            # 跨机器/自由切换：为本机所有已下载模型的 quant 自动生成 strata-<quant>.json。
            # 模型 gguf 真实存在的 quant 才会生成成功；未下载的自动跳过。
            # 这样任何用户装好模型后，前端 /api/models 与启动器都能自由切换。
            available_ups = []
            seen = set()
            for label, (low, up) in cls._QUANT_JSON_MAP.items():
                if up in seen:
                    continue
                seen.add(up)
                jp = os.path.join(install, f"strata-{low}.json")
                if os.path.isfile(jp) and cls._json_native_ok(install, low):
                    available_ups.append(up)
                    continue
                try:
                    ok, _msg = cls._refresh_json_config(label)
                    if ok and cls._json_native_ok(install, low):
                        available_ups.append(up)
                except Exception:
                    pass
            if available_ups:
                print(f"[BSAI-LLM-Strata] 可用模型配置: {', '.join(available_ups)}")
            else:
                print("[BSAI-LLM-Strata] 未检测到已下载的 Strata 模型（请先将模型 gguf 放入 "
                      "ComfyUI/models/LLM/Qwen3.8-Flash-Next/<QUANT>/ 目录）")
            # 全部 json 就绪后再选 bat 默认 quant（preferred 优先，模型文件校验通过）
            quant_upper, quant_lower = cls._pick_quant(install)
            bat = os.path.join(root, cls._LAUNCHER_BAT_NAME)
            template = (
                "@echo off\r\n"
                "chcp 65001 >nul\r\n"
                "title BSAI Strata 服务 - 双击启动\r\n"
                "echo ======================================\r\n"
                "echo    BSAI Strata LLM Service\r\n"
                "echo ======================================\r\n"
                "echo.\r\n"
                "\r\n"
                "set \"PKG=%~dp0\"\r\n"
                "set \"PY=%PKG%{PY}\"\r\n"
                "set \"STRATA=%PKG%{STRATA}\"\r\n"
                "\r\n"
                "if not exist \"%PY%\" (\r\n"
                "    echo [BSAI] 未找到 Python: %PY%\r\n"
                "    pause\r\n"
                "    exit /b 1\r\n"
                ")\r\n"
                "if not exist \"%STRATA%\" (\r\n"
                "    echo [BSAI] 未找到 Strata 运行时: %STRATA%\r\n"
                "    echo [BSAI] 请先完成 Strata 引擎安装\r\n"
                "    pause\r\n"
                "    exit /b 1\r\n"
                ")\r\n"
                "\r\n"
                "cd /d \"%STRATA%\"\r\n"
                "echo [BSAI] 正在启动 Strata 服务 ({QUANT}) ...\r\n"
                "echo [BSAI] 首次加载 125B 模型可能需要 1-3 分钟，请耐心等待\r\n"
                "\"%PY%\" \"bsai_model_manager.py\" \"{QUANT}\"\r\n"
                "if errorlevel 1 (\r\n"
                "    echo.\r\n"
                "    echo [BSAI] Strata 服务启动失败，请查看上方日志\r\n"
                "    pause\r\n"
                ")\r\n"
            ).format(PY=py_rel, STRATA=os.path.relpath(install, root), QUANT=quant_upper)
            # UTF-8 编码 + bat 首行 chcp 65001：保证中文提示不乱码
            with open(bat, "w", encoding="utf-8", errors="replace") as f:
                f.write(template)
            print(f"[BSAI-LLM-Strata] 已生成启动器 {bat}（Python={py_rel}, quant={quant_upper}；双击启动 Strata 服务；未启动则节点不自动拉起）")
            return bat
        except Exception as e:
            print(f"[BSAI-LLM-Strata] 生成 BSAI_Strata服务.bat 失败: {e}")
            return None

    # ---------- ensure running (manual launcher mode: never auto-spawn) ----------
    @classmethod
    def ensure_running(cls, wait_timeout=240.0, poll_interval=3.0, quant_label=None):
        """Block until Strata is reachable. If not running, refresh the json config
        and return a pointer to the BSAI_Strata服务.bat launcher — the engine is
        NEVER auto-spawned; it starts only when the user double-clicks the bat.

        Returns (ok, message). Called by any BSAI node before hitting /v1.
        """
        h = cls.health(timeout=1.0)
        if h.get("ok"):
            return True, "already_running"

        # Not up — keep config fresh so the launcher can start immediately,
        # then hand control to the user (manual launch only).
        label = quant_label or cls.get_preferred_quant()
        if label == "Auto (自动适配)":
            label = cls.detect_best_quant()
        ok_gen, msg_gen = cls._refresh_json_config(label)
        if not ok_gen:
            print(f"[BSAI-LLM-Strata] Warning: json config refresh failed: {msg_gen}")
        launcher = os.path.join(cls._package_root() or "", cls._LAUNCHER_BAT_NAME)
        if launcher and os.path.exists(launcher):
            print(f"[BSAI-LLM-Strata] Strata 服务未运行——请双击启动器手动启动: {launcher}")
            return False, f"Strata 服务未运行。请双击 {launcher} 启动，等待模型加载完成后再运行本节点。"
        return False, "Strata 服务未运行，且未找到 BSAI_Strata服务.bat 启动器。请检查插件安装。"

    # ---------- spawn ----------
    @classmethod
    def start(cls, run_script=None):
        """Launch a run-<quant>.bat in a detached console. Non-blocking."""
        with cls._lock:
            h = cls.health(timeout=0.5)
            if h.get("ok"):
                return {"ok": True, "already_running": True}
            scripts = cls.list_run_scripts()
            if run_script is None and scripts:
                run_script = scripts[0]
            if not run_script or not os.path.exists(run_script):
                return {"ok": False, "error": f"no run-*.bat found. Scripts: {scripts}"}
            creationflags = 0
            if sys.platform == "win32":
                # CREATE_NEW_CONSOLE 与 DETACHED_PROCESS 互斥，不能同时用（WinError 87）
                creationflags = subprocess.CREATE_NEW_CONSOLE
            try:
                cls._proc = subprocess.Popen(
                    ["cmd.exe", "/c", run_script],
                    cwd=os.path.dirname(run_script),
                    creationflags=creationflags,
                    close_fds=True,
                )
                return {"ok": True, "pid": cls._proc.pid, "script": run_script}
            except Exception as e:
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---------- auto-generate json config (fix hardcoded paths) ----------
    _QUANT_JSON_MAP = {
        "Q2_0":       ("q2_0", "Q2_0"),
        "Q2_0 (最快 / fastest)":       ("q2_0", "Q2_0"),
        "IQ2_XS":      ("iq2_xs", "IQ2_XS"),
        "IQ2_XS (略好 / better)":      ("iq2_xs", "IQ2_XS"),
        "IQ3_XXS":       ("iq3_xxs", "IQ3_XXS"),
        "IQ3_XXS (更好 / good)":       ("iq3_xxs", "IQ3_XXS"),
        "IQ3_S": ("iq3_s", "IQ3_S"),
        "IQ3_S (最好 / best quality)": ("iq3_s", "IQ3_S"),
    }

    @classmethod
    def _refresh_json_config(cls, quant_label):
        """Regenerate strata-<quant>.json with dynamically-resolved paths.
        Fixes hardcoded G:\\... paths so the plugin works on any machine."""
        install = cls.find_install()
        if not install:
            return False, "Strata install not found"

        if quant_label not in cls._QUANT_JSON_MAP:
            return False, f"unknown quant: {quant_label}"

        quant_lower, quant_upper = cls._QUANT_JSON_MAP[quant_label]
        json_path = os.path.join(install, f"strata-{quant_lower}.json")

        # Locate gguf files: use _detect_available_quants to find the actual directory
        available = cls._detect_available_quants()
        gguf_dir = None
        for q, d in available:
            if q == quant_upper:
                gguf_dir = d
                break
        if not gguf_dir:
            # Fallback to expected subdir
            gguf_dir = os.path.join(STRATA_MODEL_DIR, quant_upper)
        if not os.path.isdir(gguf_dir):
            return False, f"gguf dir not found: {gguf_dir}"
        ggufs = sorted([f for f in os.listdir(gguf_dir)
                        if f.endswith(".gguf") and not f.startswith("mmproj-")
                        and not f.endswith(".done")])
        if not ggufs:
            return False, f"no gguf files in {gguf_dir}"
        gguf1 = os.path.join(gguf_dir, ggufs[0])
        gguf2 = os.path.join(gguf_dir, ggufs[1]) if len(ggufs) >= 2 else gguf1

        # Locate mmproj
        mmproj = None
        for f in os.listdir(STRATA_MODEL_DIR):
            if f.startswith("mmproj-") and f.endswith(".gguf"):
                mmproj = os.path.join(STRATA_MODEL_DIR, f)
                break

        # Build paths
        pack_dir = os.path.join(STRATA_DATA_DIR, "packs", quant_lower)
        expert_profile = os.path.join(install, "data", "expert-profile.bin")
        mtp_dir = os.path.join(STRATA_DATA_DIR, "mtp", "rt")
        tokenizer_dir = os.path.join(pack_dir, "tokenizer")
        strata_exe = os.path.join(install, "engine", "strata.exe")
        strata_vision_exe = os.path.join(install, "engine", "strata-vision.exe")
        log_path = os.path.join(install, f"strata-{quant_lower}.log")
        # Auto-detect CUDA libraries: prefer ComfyUI's embedded Python (already has CUDA libs)
        # fallback to .venv if we installed them manually
        import sys
        cuda_lib_dirs = []
        # ComfyUI embedded Python site-packages
        comfy_site = os.path.join(sys.prefix, "Lib", "site-packages")
        for sub in ["nvidia", "nvidia", "nvidia"]:
            for ver_dir in ["cu12", "cu13", "cu11"]:
                p = os.path.join(comfy_site, sub, ver_dir, "bin", "x86_64")
                if os.path.isdir(p):
                    cuda_lib_dirs.append(p)
                    break
        # Also check .venv
        venv_cuda = os.path.join(install, ".venv", "Lib", "site-packages",
                                 "nvidia", "cu13", "bin", "x86_64")
        if os.path.isdir(venv_cuda):
            cuda_lib_dirs.append(venv_cuda)
        # Deduplicate
        cuda_lib_dirs = list(dict.fromkeys(cuda_lib_dirs))
        if not cuda_lib_dirs:
            # Fallback to expected path
            cuda_lib_dirs = [venv_cuda]

        # Read existing json to preserve user-tuned args
        cfg = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}

        # Extract tuned args from existing config (spec, kv-resident, vram-reserve, max-context)
        existing_args = cfg.get("args", [])
        def _get_arg(flag, default):
            try:
                i = existing_args.index(flag)
                if i + 1 < len(existing_args):
                    return existing_args[i + 1]
            except (ValueError, IndexError):
                pass
            return default

        spec_val = _get_arg("--spec", "8")
        kv_resident_val = _get_arg("--kv-resident", "8192")
        vram_reserve_val = _get_arg("--vram-reserve-mib", "2000")
        max_context_val = _get_arg("--max-context", "65536")
        resident_budget_val = _get_arg("--resident-budget-gib", "")

        # Auto-detect system RAM to set resident-budget-gib (leave ~8GB for OS/ComfyUI)
        if not resident_budget_val:
            try:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
                resident_budget_val = str(max(8, int(total_ram_gb - 8)))
            except Exception:
                resident_budget_val = "24"

        # Build new args with correct paths + preserved tuning
        new_args = [
            "--pack", pack_dir,
            "--native", gguf1,
            "--ple-gguf", gguf2,
            "--expert-profile", expert_profile,
            "--expert-cache", "auto",
            "--prefill", "auto",
            "--spec", spec_val, "--spec-min-p", "0.5",
            "--mtp", mtp_dir,
            "--max-context", max_context_val,
            "--kv", "q4_0",
            "--kv-resident", kv_resident_val,
            "--vram-reserve-mib", vram_reserve_val,
            "--mmap-experts",
            "--kv-grow",
            "--resident-budget-gib", resident_budget_val,
        ]

        new_cfg = {
            "fit_max_tokens": True,
            "args": new_args,
            "port": cfg.get("port", 8080),
            "vision": {
                "gpu": True,
                "model": gguf1,
                "exe": strata_vision_exe,
                "mmproj": mmproj or "",
                "max_tokens": 1024,
            },
            "cwd": install,
            "gpus_asked": True,
            "model_name": f"qwen3.8-flash-next-{quant_lower}",
            "exe": strata_exe,
            "log": log_path,
            "tokenizer": tokenizer_dir,
            "gpu": 0,
            "lib_dirs": cuda_lib_dirs,
        }

        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(new_cfg, f, indent=4, ensure_ascii=False)
            print(f"[BSAI-LLM-Strata] Auto-generated config: {json_path}")
            return True, json_path
        except Exception as e:
            return False, f"write failed: {e}"

    # ---------- one-time setup ----------
    @classmethod
    def setup(cls, quant="Q2_0", context=65536, kv="q4_0"):
        """Run setup.py against the discovered Strata install + the BSAI model dir.
        This is the equivalent of:
          python setup.py --family qwen --model <quant> \
                --gguf-dir <STRATA_MODEL_DIR> --context <context> --kv <kv> \
                --no-start --yes
        """
        install = cls.find_install()
        if not install:
            return {"ok": False, "error": "Strata source not found (no START-HERE.bat)."}
        if not os.path.isdir(STRATA_MODEL_DIR):
            return {"ok": False, "error": f"Model dir missing: {STRATA_MODEL_DIR}"}
        setup_py = os.path.join(install, "setup.py")
        if not os.path.exists(setup_py):
            return {"ok": False, "error": f"setup.py not found at {setup_py}"}
        args = [
            sys.executable, setup_py,
            "--family", "qwen",
            "--model", str(quant),
            "--gguf-dir", STRATA_MODEL_DIR,
            "--context", str(context),
            "--kv", str(kv),
            "--no-start", "--yes",
        ]
        try:
            p = subprocess.Popen(
                args, cwd=install,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            out, _ = p.communicate(timeout=600)
            return {"ok": p.returncode == 0, "returncode": p.returncode, "output": out[-4000:]}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---------- HTTP call ----------
    @staticmethod
    def chat(messages, model=DEFAULT_STRATA_MODEL, max_tokens=16384,
             temperature=0.7, timeout=600):
        url = f"{DEFAULT_STRATA_BASE_URL}/chat/completions"
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode("utf-8")
        status, raw = _blocking_post_json(
            url, body, timeout=timeout,
            headers={"Authorization": "Bearer strata"},
        )
        if status >= 400:
            raise RuntimeError(f"Strata HTTP {status}: {raw[:300]}")
        return json.loads(raw)


# ---------------------------------------------------------------------------
# NInferManager — local NInfer reverse engine
#
# The NInfer runtime (engine/ninfer-sm120|ninfer-sm89/ninfer-serve.exe, ported
# from the original ComfyUI-AIManzi-MultimodalPrompt plugin) is NOT shipped with
# any BSAI plugin — the binaries are too large for git. NInferManager.engine_dir()
# resolves it from $BSAI_NINFER_ENGINE_DIR, this plugin's own engine/, the
# upstream AIManzi plugin, or any sibling custom_nodes plugin that carries one.
# It consumes *.ninfer models from ComfyUI/models/LLM. BSAI plugins talk to it
# through the same OpenAI-compatible protocol as Strata.
#
# Port policy: NInfer listens on a DYNAMIC port starting at NINFER_PORT (8081).
# 8081 is often already taken by a user's other Strata instance (e.g. the C:\
# install), so the manager scans upward for the first free port and remembers
# it for the rest of the session. All BSAI plugins must call NInferManager
# (via ensure_running / chat / base_url) instead of hard-coding a port.
# ---------------------------------------------------------------------------
NINFER_HOST = "127.0.0.1"
NINFER_PORT = 8081
NINFER_PORT_RANGE = 10  # scan 8081..8090 for a free port
NINFER_BASE_URL = f"http://{NINFER_HOST}:{NINFER_PORT}/v1"

# ---------------------------------------------------------------------------
# NemotronManager — local llama.cpp engine for NVIDIA Nemotron 3.5 Bolt 74B-A9B
#
# 与 Strata / NInfer 并列的第三个本地 LLM 引擎选项（所有 BSAI 插件通用）：
#   - 引擎: llama.cpp (llama-server.exe, 需要 v0.7.0+，已内置 Nemotron Bolt 支持)
#   - 模型: 用户自行下载到 ComfyUI/models/LLM/Nemotron/
#           NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf (+ mmproj-*.gguf 视觉投影)
#   - 端口: 8085 (避开 Strata 8080 / NInfer 8081)
# 所有 BSAI 插件必须通过 NemotronManager (ensure_running / chat / base_url)
# 调用，而不是硬编码端口。
# ---------------------------------------------------------------------------
NEMOTRON_HOST = "127.0.0.1"
NEMOTRON_PORT = 8085
NEMOTRON_BASE_URL = f"http://{NEMOTRON_HOST}:{NEMOTRON_PORT}/v1"
NEMOTRON_MODEL_DIR = os.path.join(folder_paths.models_dir, "LLM", "Nemotron")
NEMOTRON_DEFAULT_ALIAS = "Nemotron-3.5-Bolt-74B-A9B"
# 提示词增强场景上下文需求很小（system+user 几百 token，输出几百 token），
# 64K 上下文会让 KV 缓存占满显存被迫走 CPU，速度暴跌。8K 足够且释放大量显存。
# 可用环境变量 BSAI_NEMOTRON_CTX 覆盖。
NEMOTRON_DEFAULT_CONTEXT = int(os.environ.get("BSAI_NEMOTRON_CTX", "8192"))

# ---------------------------------------------------------------------------
# Flint 40B：与 Strata / NInfer / Nemotron 并列的第四个本地 LLM 引擎选项。
#   - 引擎: llama.cpp (llama-server.exe)
#   - 模型: 用户自行下载到 ComfyUI/models/LLM/flint/  (*.gguf)
#   - 端口: 8082 (避开 Strata 8080 / NInfer 8081 / Nemotron 8085)
# 所有 BSAI 插件必须通过 FlintManager (ensure_running / chat / base_url) 调用。
# ---------------------------------------------------------------------------
FLINT_HOST = "127.0.0.1"
FLINT_PORT = 8082
FLINT_BASE_URL = f"http://{FLINT_HOST}:{FLINT_PORT}/v1"
FLINT_MODEL_DIR = os.path.join(folder_paths.models_dir, "LLM", "flint")
FLINT_DEFAULT_ALIAS = "Flint-40B"
# Flint 主打长上下文编程，默认 64K；可用环境变量 BSAI_FLINT_CTX 覆盖。
FLINT_DEFAULT_CONTEXT = int(os.environ.get("BSAI_FLINT_CTX", "65536"))

# ComfyUI custom_nodes root: this file lives in <root>/BSAI-LLM-Strata/
_CUSTOM_NODES_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NInferManager:
    """Auto-discover the bundled ninfer-serve.exe + first *.ninfer model,
    start it with --vision on a free local port, and health-check /v1/models."""

    _lock = threading.Lock()
    _proc = None
    _detected_engine = None
    _detected_model = None
    _model_ids = []
    _active_port = None  # resolved free port for this session
    _active_model_path = None  # absolute path of the model currently loaded

    # ---------- port resolution ----------
    @classmethod
    def base_url(cls):
        """URL of the session's NInfer server (dynamic port)."""
        if cls._active_port:
            return f"http://{NINFER_HOST}:{cls._active_port}/v1"
        return NINFER_BASE_URL

    @staticmethod
    def _port_free(port, timeout=0.3):
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((NINFER_HOST, port))
            s.close()
            return False  # something is listening -> occupied
        except Exception:
            return True

    @classmethod
    def _resolve_free_port(cls):
        if cls._active_port and cls._port_free(cls._active_port):
            return cls._active_port
        for p in range(NINFER_PORT, NINFER_PORT + NINFER_PORT_RANGE):
            if cls._port_free(p):
                cls._active_port = p
                return p
        cls._active_port = NINFER_PORT
        return NINFER_PORT

    # ---------- discovery ----------
    @classmethod
    def engine_dir(cls):
        """Return the directory holding ninfer-sm<XX>/ + runtime/, or None.

        The engine binaries are too large for git, so they are not redistributed
        with any BSAI plugin. Look for them in order:
          1) $BSAI_NINFER_ENGINE_DIR
          2) BSAI-FLUX3-ComfyUI/engine/   (bundled copy, when present)
          3) ComfyUI-AIManzi-MultimodalPrompt/engine/  (upstream source)
          4) any custom_nodes/*/engine/ that already contains ninfer-serve.exe
        """
        cands = []
        env_dir = os.environ.get("BSAI_NINFER_ENGINE_DIR", "").strip()
        if env_dir:
            cands.append(env_dir)
        cands.append(os.path.join(_CUSTOM_NODES_ROOT, "BSAI-FLUX3-ComfyUI", "engine"))
        cands.append(os.path.join(_CUSTOM_NODES_ROOT,
                                  "ComfyUI-AIManzi-MultimodalPrompt", "engine"))
        try:
            for name in sorted(os.listdir(_CUSTOM_NODES_ROOT)):
                cands.append(os.path.join(_CUSTOM_NODES_ROOT, name, "engine"))
        except OSError:
            pass
        for d in cands:
            for sub in ("ninfer-sm120", "ninfer-sm89"):
                if os.path.exists(os.path.join(d, sub, "ninfer-serve.exe")):
                    return d
        return None

    @classmethod
    def find_engine(cls):
        """Return the ninfer-serve.exe path for this GPU, or None."""
        if cls._detected_engine:
            return cls._detected_engine
        engine_root = cls.engine_dir()
        if not engine_root:
            return None
        # Pick sm120 for RTX 50-series, sm89 for RTX 40-series
        profile = None
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            cap = float(out.stdout.strip().splitlines()[0])
            if cap >= 12.0:
                profile = "ninfer-sm120"
            elif cap >= 8.9:
                profile = "ninfer-sm89"
        except Exception:
            pass
        if not profile:
            for cand in ("ninfer-sm120", "ninfer-sm89"):
                exe = os.path.join(engine_root, cand, "ninfer-serve.exe")
                if os.path.exists(exe):
                    profile = cand
                    break
        if not profile:
            return None
        exe = os.path.join(engine_root, profile, "ninfer-serve.exe")
        if os.path.exists(exe):
            cls._detected_engine = exe
            return exe
        return None

    @classmethod
    def find_model(cls):
        """Return the first *.ninfer model under ComfyUI/models/LLM, or None."""
        if cls._detected_model:
            return cls._detected_model
        lst = cls.list_models()
        if not lst:
            return None
        cls._detected_model = lst[0][1]
        return cls._detected_model

    @classmethod
    def list_models(cls):
        """Return [(display_rel_path, absolute_path), ...] for every *.ninfer
        model under ComfyUI/models/LLM (recursive, sorted)."""
        out = []
        llm_root = os.path.join(folder_paths.models_dir, "LLM")
        if not os.path.isdir(llm_root):
            return out
        for root, _dirs, files in os.walk(llm_root):
            for f in sorted(files):
                if f.lower().endswith(".ninfer"):
                    abs_p = os.path.join(root, f)
                    rel = os.path.relpath(abs_p, llm_root).replace("\\", "/")
                    out.append((rel, abs_p))
        return out

    @classmethod
    def resolve_model(cls, model_path):
        """Accept a display rel path or absolute path; return absolute path."""
        if not model_path:
            return cls.find_model()
        p = str(model_path).replace("/", os.sep)
        if os.path.isabs(p):
            return p if os.path.exists(p) else None
        cand = os.path.join(folder_paths.models_dir, "LLM", p)
        return cand if os.path.exists(cand) else None

    @classmethod
    def _stop(cls, timeout=15.0):
        """Terminate the running ninfer-serve process (used for model switch)."""
        proc = cls._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout)
                except Exception:
                    pass
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    import subprocess as _sp
                    _sp.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True, timeout=10)
                except Exception:
                    pass
        cls._proc = None
        cls._model_ids = []

    # ---------- health ----------
    @classmethod
    def health(cls, timeout=1.5):
        url = f"{cls.base_url()}/models"
        try:
            status, body = _blocking_get_json(url, timeout=timeout)
            data = json.loads(body)
            cls._model_ids = [m.get("id", "?") for m in data.get("data", [])]
            return {"ok": True, "models": cls._model_ids}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @classmethod
    def models(cls):
        """Return model ids served by the (running) NInfer server."""
        h = cls.health(timeout=1.0)
        return h.get("models") or cls._model_ids

    # ---------- ensure running (auto-start + wait) ----------
    @classmethod
    def ensure_running(cls, wait_timeout=180.0, poll_interval=2.0, model_path=None):
        """Block until our NInfer server is reachable; spawn it if needed.
        model_path: optional absolute (or LLM-relative) path to a specific
        .ninfer model; switching to a different model restarts the engine."""
        target = cls.resolve_model(model_path)
        if target is None:
            target = cls.find_model()
        # If we already own a port with a live service and the same model, done.
        if (cls._active_port and cls._active_model_path == target
                and cls.health(timeout=1.0).get("ok")):
            return True, f"already_running @{cls._active_port} ({os.path.basename(target)})"
        with cls._lock:
            # Re-probe under the lock (avoid double spawn).
            if (cls._active_port and cls._active_model_path == target
                    and cls.health(timeout=1.0).get("ok")):
                return True, f"already_running @{cls._active_port} ({os.path.basename(target)})"
            engine = cls.find_engine()
            if not engine:
                return False, ("ninfer-serve.exe 未找到。NInfer 引擎体积过大，不随插件一起分发："
                               "请安装 ComfyUI-AIManzi-MultimodalPrompt 插件（自带 engine/），"
                               "或把 engine/ 目录放到 custom_nodes/BSAI-FLUX3-ComfyUI/engine，"
                               "或设置环境变量 BSAI_NINFER_ENGINE_DIR 指向 engine 目录。")
            model = target or cls.find_model()
            if not model:
                return False, "未找到 *.ninfer 模型。请把 .ninfer 模型放入 ComfyUI/models/LLM。"
            # Switching models: stop the current engine first.
            if cls._proc is not None and cls._proc.poll() is None:
                if cls._active_model_path != model:
                    print(f"[BSAI-LLM-Strata] NInfer 切换模型: "
                          f"{os.path.basename(cls._active_model_path or '?')} -> {os.path.basename(model)}")
                    cls._stop()
                    time.sleep(2)
            port = cls._resolve_free_port()
            # Kill any leftover ninfer from a previous ComfyUI session that is
            # still holding VRAM (otherwise the new launch fails with OOM).
            cls._kill_orphans()
            cls._release_comfy_vram()
            command = [
                engine, model,
                "--host", NINFER_HOST, "--port", str(port),
                "--max-context", "32768", "--kv-capacity", "32768",
                "--max-concurrency", "1", "--kv-dtype", "fp8",
                "--default-max-tokens", "32768",
                "--vision", "--no-thinking",
            ]
            cls._active_model_path = model
            engine_root = cls.engine_dir() or ""
            env = os.environ.copy()
            env["PATH"] = (os.path.dirname(engine) + os.pathsep
                           + os.path.join(engine_root, "runtime") + os.pathsep
                           + env.get("PATH", ""))
            log_path = os.path.join(_PLUGIN_DIR, "ninfer-startup.log")
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} BSAI start @{port} "
                             f"model={os.path.basename(model)} ---\n")
                    cls._proc = subprocess.Popen(
                        command, cwd=engine_root or os.path.dirname(engine), env=env,
                        stdout=lf, stderr=subprocess.STDOUT,
                    )
            except Exception as e:
                return False, f"启动 ninfer 失败: {type(e).__name__}: {e}"
            deadline = time.time() + wait_timeout
            while time.time() < deadline:
                time.sleep(poll_interval)
                h = cls.health(timeout=2.0)
                if h.get("ok"):
                    print(f"[BSAI-LLM-Strata] NInfer 引擎就绪 @{port}: {h.get('models')}")
                    return True, f"started_and_ready @{port}"
                if cls._proc.poll() is not None:
                    try:
                        tail = open(log_path, encoding="utf-8", errors="replace").read().splitlines()[-15:]
                    except Exception:
                        tail = []
                    return False, "NInfer 进程已退出。引擎日志:\n" + "\n".join(tail)
            return False, f"等待 NInfer 启动超时({wait_timeout:.0f}s)"

    @staticmethod
    def _kill_orphans():
        """Kill any leftover ninfer-serve.exe from our engine directory.

        ComfyUI restarts don't terminate child processes; a previous session's
        ninfer can still hold 7-15 GB VRAM and cause the next launch to fail
        with 'model weights require X bytes but only Y free'. We only kill
        processes whose exe lives under our engine dir, never user services.
        """
        try:
            import psutil  # type: ignore
        except ImportError:
            return
        engine_root = NInferManager.engine_dir()
        if not engine_root:
            return
        killed = []
        for proc in psutil.process_iter(["name", "exe"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name != "ninfer-serve.exe":
                    continue
                exe = proc.info.get("exe") or ""
                if engine_root.lower() in exe.lower():
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except Exception:
                        proc.kill()
                        proc.wait(timeout=10)
                    killed.append(proc.pid)
            except Exception:
                continue
        if killed:
            print(f"[BSAI-LLM-Strata] NInfer orphan processes killed: {killed}")
            time.sleep(1.0)

    @staticmethod
    def _release_comfy_vram():
        """Unload ComfyUI models so the ~16GB ninfer runtime can load."""
        try:
            import comfy.model_management as mm
            mm.unload_all_models()
            try:
                mm.soft_empty_cache(True)
            except TypeError:
                mm.soft_empty_cache()
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    # ---------- HTTP call ----------
    @staticmethod
    def chat(messages, model=None, max_tokens=8192, temperature=0.7, timeout=600):
        """POST /v1/chat/completions to the local NInfer server (OpenAI-compatible)."""
        if not model:
            mids = NInferManager.models()
            model = mids[0] if mids else "default"
        url = f"{NInferManager.base_url()}/chat/completions"
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode("utf-8")
        status, raw = _blocking_post_json(url, body, timeout=timeout)
        if status >= 400:
            raise RuntimeError(f"NInfer HTTP {status}: {raw[:300]}")
        return json.loads(raw)


# ---------------------------------------------------------------------------
# NemotronManager — local llama.cpp engine for NVIDIA Nemotron 3.5 Bolt 74B-A9B
#
# 与 Strata / NInfer 并列的第三个本地 LLM 引擎选项。llama.cpp (v0.7.0+) 已内置
# Nemotron Bolt / Nemotron Super 3.5 VL 支持；模型 GGUFs 由用户自行下载到
# ComfyUI/models/LLM/Nemotron/。本管理器自动发现 llama-server.exe 与模型文件，
# 在 127.0.0.1:8085 启动 OpenAI 兼容服务。
# ---------------------------------------------------------------------------
class NemotronManager:
    """Auto-discover llama-server.exe + Nemotron GGUF model, start it on
    127.0.0.1:8085 and health-check /v1/models. OpenAI-compatible protocol,
    same as Strata / NInfer — any BSAI node can route to it."""

    _lock = threading.Lock()
    _proc = None
    _detected_engine = None
    _detected_model = None
    _detected_mmproj = None
    _model_ids = []
    _active_model_path = None

    # ---------- discovery ----------
    @classmethod
    def llama_dir(cls):
        """Directory that contains llama-server.exe (bundled, next to models,
        or the shared custom_nodes/BSAI-llama.cpp runtime)."""
        bundled = os.path.join(_PLUGIN_DIR, "Nemotron", "llama.cpp")
        if os.path.isdir(bundled):
            return bundled
        model_side = os.path.join(NEMOTRON_MODEL_DIR, "llama.cpp")
        if os.path.isdir(model_side):
            return model_side
        # 整合包共享 llama.cpp：custom_nodes/BSAI-llama.cpp 或 custom_nodes/llama.cpp
        for rel in ("BSAI-llama.cpp", "llama.cpp"):
            cand = os.path.join(_CUSTOM_NODES_ROOT, rel)
            if os.path.isdir(cand):
                return cand
        return bundled

    @classmethod
    def find_engine(cls):
        """Return the llama-server.exe path, or None."""
        if cls._detected_engine:
            return cls._detected_engine
        exe = os.path.join(cls.llama_dir(), "llama-server.exe")
        if os.path.exists(exe):
            cls._detected_engine = exe
            return exe
        return None

    @classmethod
    def list_models(cls):
        """Return [(display_rel_path, absolute_path, kind), ...] for every
        *.gguf under models/LLM/Nemotron. kind: 'model' or 'mmproj'."""
        out = []
        if not os.path.isdir(NEMOTRON_MODEL_DIR):
            return out
        for root, _dirs, files in os.walk(NEMOTRON_MODEL_DIR):
            for f in sorted(files):
                if not f.lower().endswith(".gguf"):
                    continue
                low = f.lower()
                abs_p = os.path.join(root, f)
                rel = os.path.relpath(abs_p, NEMOTRON_MODEL_DIR).replace("\\", "/")
                if "mmproj" in low:
                    out.append((rel, abs_p, "mmproj"))
                else:
                    out.append((rel, abs_p, "model"))
        return out

    @classmethod
    def find_model(cls):
        """Return the first non-mmproj GGUF (the main model), or None."""
        if cls._detected_model:
            return cls._detected_model
        for rel, abs_p, kind in cls.list_models():
            if kind == "model":
                cls._detected_model = abs_p
                return abs_p
        return None

    @classmethod
    def find_mmproj(cls):
        """Return the mmproj-*.gguf vision projector, or None."""
        if cls._detected_mmproj:
            return cls._detected_mmproj
        for rel, abs_p, kind in cls.list_models():
            if kind == "mmproj":
                cls._detected_mmproj = abs_p
                return abs_p
        return None

    # ---------- health ----------
    @classmethod
    def health(cls, timeout=1.5):
        url = f"http://{NEMOTRON_HOST}:{NEMOTRON_PORT}/v1/models"
        try:
            status, body = _blocking_get_json(url, timeout=timeout)
            data = json.loads(body)
            cls._model_ids = [m.get("id", "?") for m in data.get("data", [])]
            return {"ok": True, "models": cls._model_ids}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @classmethod
    def models(cls):
        """Return model ids served by the (running) Nemotron server."""
        h = cls.health(timeout=1.0)
        return h.get("models") or cls._model_ids

    @classmethod
    def base_url(cls):
        return NEMOTRON_BASE_URL

    @classmethod
    def resolve_model(cls, model_path):
        """Accept a display rel path or absolute path; return absolute path."""
        if not model_path:
            return cls.find_model()
        p = str(model_path).replace("/", os.sep)
        if os.path.isabs(p):
            return p if os.path.exists(p) else None
        cand = os.path.join(NEMOTRON_MODEL_DIR, p)
        return cand if os.path.exists(cand) else None

    # ---------- ensure running (auto-start + wait) ----------
    @classmethod
    def ensure_running(cls, wait_timeout=300.0, poll_interval=2.0, model_path=None):
        """Block until the Nemotron llama.cpp server is reachable on 8085;
        spawn llama-server.exe if not running (auto-detect model + mmproj).
        model_path: optional rel/abs GGUF path; switching restarts the server."""
        target = cls.resolve_model(model_path)
        if target is None:
            target = cls.find_model()
        # If already serving the requested model, done.
        _h = cls.health(timeout=1.0)
        if (cls._proc is not None and cls._proc.poll() is None
                and cls._active_model_path == target
                and _h.get("ok") and _h.get("models")):
            return True, f"already_running ({os.path.basename(target or '?')})"
        with cls._lock:
            _h2 = cls.health(timeout=1.0)
            if (cls._proc is not None and cls._proc.poll() is None
                    and cls._active_model_path == target
                    and _h2.get("ok") and _h2.get("models")):
                return True, f"already_running ({os.path.basename(target or '?')})"
            engine = cls.find_engine()
            if not engine:
                return False, ("llama-server.exe 未找到。请把 llama.cpp 运行时放入 "
                               "BSAI-LLM-Strata/Nemotron/llama.cpp/（或 models/LLM/Nemotron/llama.cpp/）。")
            model = target or cls.find_model()
            if not model:
                return False, ("未找到 Nemotron GGUF 模型。请把模型下载到 "
                               "ComfyUI/models/LLM/Nemotron/（例如 NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf）。")
            mmproj = cls.find_mmproj()
            # Switching model / restart: stop the current server first.
            if cls._proc is not None and cls._proc.poll() is None:
                if cls._active_model_path != model:
                    print(f"[BSAI-LLM-Strata] Nemotron 切换模型: "
                          f"{os.path.basename(cls._active_model_path or '?')} -> {os.path.basename(model)}")
                    cls._stop()
                    time.sleep(2)
            command = [
                engine,
                "-m", model,
                "--host", NEMOTRON_HOST,
                "--port", str(NEMOTRON_PORT),
                "-c", str(NEMOTRON_DEFAULT_CONTEXT),
                "-ngl", "99",
                # Flash Attention：RTX 50 系（Blackwell）上自动选择最快实现
                "-fa", "on",
                # KV 缓存 8bit 量化：KV 显存减半、带宽压力大降，提速最明显
                "-ctk", "q8_0",
                "-ctv", "q8_0",
                # 更大的批/微批大小加速 prompt 预填充阶段
                "-b", "4096",
                "-ub", "2048",
                "--jinja",
                "--parallel", "1",
                "--alias", NEMOTRON_DEFAULT_ALIAS,
            ]
            if mmproj:
                command += ["--mmproj", mmproj]
            cls._active_model_path = model
            env = os.environ.copy()
            env["PATH"] = (os.path.dirname(engine) + os.pathsep + env.get("PATH", ""))
            log_path = os.path.join(_PLUGIN_DIR, "nemotron-startup.log")
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} BSAI Nemotron start @{NEMOTRON_PORT} "
                             f"model={os.path.basename(model)} "
                             f"mmproj={os.path.basename(mmproj) if mmproj else 'none'} ---\n")
                    cls._proc = subprocess.Popen(command, cwd=os.path.dirname(engine), env=env,
                                                 stdout=lf, stderr=subprocess.STDOUT)
            except Exception as e:
                return False, f"启动 llama-server 失败: {type(e).__name__}: {e}"
            deadline = time.time() + wait_timeout
            while time.time() < deadline:
                time.sleep(poll_interval)
                h = cls.health(timeout=2.0)
                if h.get("ok") and h.get("models"):
                    print(f"[BSAI-LLM-Strata] Nemotron 引擎就绪 @{NEMOTRON_PORT}: {h.get('models')}")
                    return True, f"started_and_ready @{NEMOTRON_PORT}"
                if cls._proc.poll() is not None:
                    try:
                        tail = open(log_path, encoding="utf-8", errors="replace").read().splitlines()[-15:]
                    except Exception:
                        tail = []
                    return False, "llama-server 进程已退出。引擎日志:\n" + "\n".join(tail)
            return False, f"等待 Nemotron 启动超时({wait_timeout:.0f}s)。请查看 nemotron-startup.log。"

    @classmethod
    def _stop(cls, timeout=15.0):
        """Terminate the running llama-server process."""
        proc = cls._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout)
                except Exception:
                    pass
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True, timeout=10)
                except Exception:
                    pass
        cls._proc = None
        cls._model_ids = []

    @classmethod
    def stop(cls):
        cls._stop()
        return {"ok": True}

    # ---------- HTTP call ----------
    @staticmethod
    def chat(messages, model=None, max_tokens=4096, temperature=0.7, timeout=600):
        """POST /v1/chat/completions to the local Nemotron server (OpenAI-compatible)."""
        if not model:
            mids = NemotronManager.models()
            model = mids[0] if mids else NEMOTRON_DEFAULT_ALIAS
        url = f"{NEMOTRON_BASE_URL}/chat/completions"
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode("utf-8")
        status, raw = _blocking_post_json(url, body, timeout=timeout)
        if status >= 400:
            raise RuntimeError(f"Nemotron HTTP {status}: {raw[:300]}")
        return json.loads(raw)


class FlintManager:
    """Auto-discover llama-server.exe + Flint GGUF model, start it on
    127.0.0.1:8082 and health-check /v1/models. OpenAI-compatible protocol,
    same as Strata / NInfer / Nemotron — any BSAI node can route to it."""

    _lock = threading.Lock()
    _proc = None
    _detected_engine = None
    _detected_model = None
    _model_ids = []
    _active_model_path = None

    # ---------- discovery ----------
    @classmethod
    def llama_dir(cls):
        """Directory that contains llama-server.exe.

        Lookup order:
          1) BSAI-LLM-Strata/flint/llama.cpp/        (bundled with this plugin)
          2) models/LLM/flint/llama.cpp/              (next to the model files)
          3) custom_nodes/BSAI-llama.cpp/             (整合包共享 llama.cpp)
          4) custom_nodes/llama.cpp/
          5) <integration root>/BSAI-llama.cpp/
          6) <integration root>/llama.cpp/
        """
        bundled = os.path.join(_PLUGIN_DIR, "flint", "llama.cpp")
        if os.path.isdir(bundled):
            return bundled
        model_side = os.path.join(FLINT_MODEL_DIR, "llama.cpp")
        if os.path.isdir(model_side):
            return model_side
        # 整合包共享 llama.cpp：custom_nodes/BSAI-llama.cpp 或 custom_nodes/llama.cpp
        for rel in ("BSAI-llama.cpp", "llama.cpp"):
            cand = os.path.join(_CUSTOM_NODES_ROOT, rel)
            if os.path.isdir(cand):
                return cand
        # 整合包根：本文件位于 <root>/ComfyUI/custom_nodes/BSAI-LLM-Strata/strata_manager.py
        integration_root = os.path.dirname(os.path.dirname(os.path.dirname(_PLUGIN_DIR)))
        for rel in ("BSAI-llama.cpp", "llama.cpp"):
            cand = os.path.join(integration_root, rel)
            if os.path.isdir(cand):
                return cand
        return bundled

    @classmethod
    def find_engine(cls):
        """Return the llama-server.exe path, or None."""
        if cls._detected_engine:
            return cls._detected_engine
        exe = os.path.join(cls.llama_dir(), "llama-server.exe")
        if os.path.exists(exe):
            cls._detected_engine = exe
            return exe
        return None

    @classmethod
    def list_models(cls):
        """Return [(display_rel_path, absolute_path), ...] for every *.gguf
        model under models/LLM/flint (recursive, sorted)."""
        out = []
        if not os.path.isdir(FLINT_MODEL_DIR):
            return out
        for root, _dirs, files in os.walk(FLINT_MODEL_DIR):
            for f in sorted(files):
                if not f.lower().endswith(".gguf"):
                    continue
                if "mmproj" in f.lower():
                    continue
                abs_p = os.path.join(root, f)
                rel = os.path.relpath(abs_p, FLINT_MODEL_DIR).replace("\\", "/")
                out.append((rel, abs_p))
        return out

    @classmethod
    def find_model(cls):
        """Return the first GGUF model, or None."""
        if cls._detected_model:
            return cls._detected_model
        for rel, abs_p in cls.list_models():
            cls._detected_model = abs_p
            return abs_p
        return None

    # ---------- health ----------
    @classmethod
    def health(cls, timeout=1.5):
        url = f"http://{FLINT_HOST}:{FLINT_PORT}/v1/models"
        try:
            status, body = _blocking_get_json(url, timeout=timeout)
            data = json.loads(body)
            cls._model_ids = [m.get("id", "?") for m in data.get("data", [])]
            return {"ok": True, "models": cls._model_ids}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @classmethod
    def models(cls):
        """Return model ids served by the (running) Flint server."""
        h = cls.health(timeout=1.0)
        return h.get("models") or cls._model_ids

    @classmethod
    def base_url(cls):
        return FLINT_BASE_URL

    @classmethod
    def resolve_model(cls, model_path):
        """Accept a display rel path or absolute path; return absolute path."""
        if not model_path:
            return cls.find_model()
        p = str(model_path).replace("/", os.sep)
        if os.path.isabs(p):
            return p if os.path.exists(p) else None
        cand = os.path.join(FLINT_MODEL_DIR, p)
        return cand if os.path.exists(cand) else None

    # ---------- ensure running (auto-start + wait) ----------
    @classmethod
    def ensure_running(cls, wait_timeout=300.0, poll_interval=2.0, model_path=None):
        """Block until the Flint llama.cpp server is reachable on 8082;
        spawn llama-server.exe if not running.
        model_path: optional rel/abs GGUF path; switching restarts the server."""
        target = cls.resolve_model(model_path)
        if target is None:
            target = cls.find_model()
        _h = cls.health(timeout=1.0)
        if (cls._proc is not None and cls._proc.poll() is None
                and cls._active_model_path == target
                and _h.get("ok") and _h.get("models")):
            return True, f"already_running ({os.path.basename(target or '?')})"
        with cls._lock:
            _h2 = cls.health(timeout=1.0)
            if (cls._proc is not None and cls._proc.poll() is None
                    and cls._active_model_path == target
                    and _h2.get("ok") and _h2.get("models")):
                return True, f"already_running ({os.path.basename(target or '?')})"
            engine = cls.find_engine()
            if not engine:
                return False, ("llama-server.exe 未找到。请把 llama.cpp 运行时放入 "
                               "BSAI-LLM-Strata/flint/llama.cpp/、models/LLM/flint/llama.cpp/，"
                               "或整合包根的 BSAI-llama.cpp/ / llama.cpp/。")
            model = target or cls.find_model()
            if not model:
                return False, ("未找到 Flint GGUF 模型。请把模型下载到 "
                               "ComfyUI/models/LLM/flint/（例如 flint-UD-Q4_K_XL.gguf）。")
            if cls._proc is not None and cls._proc.poll() is None:
                if cls._active_model_path != model:
                    print(f"[BSAI-LLM-Strata] Flint 切换模型: "
                          f"{os.path.basename(cls._active_model_path or '?')} -> {os.path.basename(model)}")
                    cls._stop()
                    time.sleep(2)
            command = [
                engine,
                "-m", model,
                "--host", FLINT_HOST,
                "--port", str(FLINT_PORT),
                "-c", str(FLINT_DEFAULT_CONTEXT),
                "-ngl", "99",
                "-fa", "auto",
                "--jinja",
                "--parallel", "1",
                "--alias", FLINT_DEFAULT_ALIAS,
                "--temp", "0.6", "--top-p", "0.95", "--top-k", "20",
                "--repeat-penalty", "1.05",
                "-n", str(FLINT_DEFAULT_CONTEXT),
            ]
            cls._active_model_path = model
            env = os.environ.copy()
            env["PATH"] = (os.path.dirname(engine) + os.pathsep + env.get("PATH", ""))
            log_path = os.path.join(_PLUGIN_DIR, "flint-startup.log")
            try:
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} BSAI Flint start @{FLINT_PORT} "
                             f"model={os.path.basename(model)} ---\n")
                    cls._proc = subprocess.Popen(command, cwd=os.path.dirname(engine), env=env,
                                                 stdout=lf, stderr=subprocess.STDOUT)
            except Exception as e:
                return False, f"启动 llama-server 失败: {type(e).__name__}: {e}"
            deadline = time.time() + wait_timeout
            while time.time() < deadline:
                time.sleep(poll_interval)
                h = cls.health(timeout=2.0)
                if h.get("ok") and h.get("models"):
                    print(f"[BSAI-LLM-Strata] Flint 引擎就绪 @{FLINT_PORT}: {h.get('models')}")
                    return True, f"started_and_ready @{FLINT_PORT}"
                if cls._proc.poll() is not None:
                    try:
                        tail = open(log_path, encoding="utf-8", errors="replace").read().splitlines()[-15:]
                    except Exception:
                        tail = []
                    return False, "llama-server 进程已退出。引擎日志:\n" + "\n".join(tail)
            return False, f"等待 Flint 启动超时({wait_timeout:.0f}s)。请查看 flint-startup.log。"

    @classmethod
    def _stop(cls, timeout=15.0):
        """Terminate the running llama-server process."""
        proc = cls._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout)
                except Exception:
                    pass
            except Exception:
                pass
            if proc.poll() is None:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                   capture_output=True, timeout=10)
                except Exception:
                    pass
        cls._proc = None
        cls._model_ids = []

    @classmethod
    def stop(cls):
        cls._stop()
        return {"ok": True}

    # ---------- HTTP call ----------
    @staticmethod
    def chat(messages, model=None, max_tokens=4096, temperature=0.7, timeout=600):
        """POST /v1/chat/completions to the local Flint server (OpenAI-compatible)."""
        if not model:
            mids = FlintManager.models()
            model = mids[0] if mids else FLINT_DEFAULT_ALIAS
        url = f"{FLINT_BASE_URL}/chat/completions"
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }).encode("utf-8")
        status, raw = _blocking_post_json(url, body, timeout=timeout)
        if status >= 400:
            raise RuntimeError(f"Flint HTTP {status}: {raw[:300]}")
        return json.loads(raw)


# ---------------------------------------------------------------------------
# ComfyUI nodes
# ---------------------------------------------------------------------------
class BSAI_LLM_Call:
    """One-shot chat completion via local Strata or NInfer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (["BSAI Strata (Qwen3.8-Flash-Next @8080)",
                            "BSAI Nemotron (llama.cpp @8085)",
                            "BSAI NInfer (本地反推 @8081)"],
                           {"default": "BSAI Strata (Qwen3.8-Flash-Next @8080)",
                            "tooltip": "Engine: Strata (125B MoE, GGUF) / Nemotron (llama.cpp, 74B-A9B) / NInfer (.ninfer vision model)"}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                      "tooltip": "User prompt / 用户提示词"}),
                "system": ("STRING", {"multiline": True, "default": "You are a helpful assistant.",
                                     "tooltip": "System prompt / 系统提示词"}),
                "quant": (list(QUANT_OPTIONS.keys()), {"default": DEFAULT_QUANT_LABEL,
                           "tooltip": "Strata engine only: which quant to auto-start (if not running)"}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 32768, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "model": ("STRING", {"default": "",
                                    "tooltip": "Optional model name override (default: server-reported id)"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "BSAI/LLM"
    DESCRIPTION = "Call local Qwen3.8-Flash-Next via Strata, or local NInfer (OpenAI-compatible)"

    def execute(self, engine, prompt, system, quant, max_tokens, temperature, model=""):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if str(engine).startswith("BSAI Strata"):
            tag = "Strata"
        elif str(engine).startswith("BSAI Nemotron"):
            tag = "Nemotron"
        else:
            tag = "NInfer"
        try:
            if tag == "Strata":
                ok, msg = StrataManager.ensure_running(quant_label=quant)
                if not ok:
                    return (f"[BSAI-LLM-Strata] Strata 启动失败: {msg}",)
                mid = model or DEFAULT_STRATA_MODEL
                r = StrataManager.chat(msgs, model=mid, max_tokens=max_tokens,
                                       temperature=temperature)
            elif tag == "Nemotron":
                ok, msg = NemotronManager.ensure_running()
                if not ok:
                    return (f"[BSAI-LLM-Strata] Nemotron 启动失败: {msg}",)
                r = NemotronManager.chat(msgs, model=model or None, max_tokens=max_tokens,
                                         temperature=temperature)
            else:
                ok, msg = NInferManager.ensure_running()
                if not ok:
                    return (f"[BSAI-LLM-Strata] NInfer 启动失败: {msg}",)
                r = NInferManager.chat(msgs, model=model or None, max_tokens=max_tokens,
                                       temperature=temperature)
            text = r["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"[BSAI-LLM-Strata {tag} ERROR] {type(e).__name__}: {e}\n" \
                   f"Is the engine running? Strata: /bsai/strata/health, Nemotron: /bsai/nemotron/health, NInfer: /bsai/ninfer/health."
        return (text,)


class BSAI_LLM_VisionDescribe:
    """Image -> text description. Sends base64 data URL to Strata mmproj or NInfer vision."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (["BSAI Strata (Qwen3.8-Flash-Next @8080)",
                            "BSAI Nemotron (llama.cpp @8085)",
                            "BSAI NInfer (本地反推 @8081)"],
                           {"default": "BSAI Strata (Qwen3.8-Flash-Next @8080)",
                            "tooltip": "Engine: Strata (mmproj) / Nemotron (llama.cpp --mmproj) / NInfer (--vision)"}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image in detail, in Chinese.",
                                     "tooltip": "Question/description prompt"}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 64}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("description",)
    FUNCTION = "execute"
    CATEGORY = "BSAI/LLM"
    DESCRIPTION = "Vision: ask Qwen3.8-Flash-Next / NInfer about an input image"

    def execute(self, engine, image, prompt, max_tokens):
        import base64
        import io
        import numpy as np
        from PIL import Image as PILImage

        # ComfyUI IMAGE is [B,H,W,C] float0..1; take frame 0
        arr = (np.clip(image[0].cpu().numpy(), 0.0, 1.0) * 255).astype(np.uint8)
        pil = PILImage.fromarray(arr)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"

        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]
        if str(engine).startswith("BSAI Strata"):
            tag = "Strata"
        elif str(engine).startswith("BSAI Nemotron"):
            tag = "Nemotron"
        else:
            tag = "NInfer"
        try:
            if tag == "Strata":
                StrataManager.ensure_running()
                r = StrataManager.chat(msgs, max_tokens=max_tokens, temperature=0.2)
            elif tag == "Nemotron":
                NemotronManager.ensure_running()
                r = NemotronManager.chat(msgs, max_tokens=max_tokens, temperature=0.2)
            else:
                NInferManager.ensure_running()
                r = NInferManager.chat(msgs, max_tokens=max_tokens, temperature=0.2)
            text = r["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"[BSAI-LLM-Strata {tag} Vision ERROR] {type(e).__name__}: {e}"
        return (text,)


class BSAI_LLM_PromptEnhance:
    """Wrap a rough prompt into a richer cinematic prompt via local Strata or NInfer."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "engine": (["BSAI Strata (Qwen3.8-Flash-Next @8080)",
                            "BSAI Nemotron (llama.cpp @8085)",
                            "BSAI NInfer (本地反推 @8081)"],
                           {"default": "BSAI Strata (Qwen3.8-Flash-Next @8080)",
                            "tooltip": "Engine: Strata / Nemotron / NInfer"}),
                "raw_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "style": (["cinematic", "anime", "photoreal", "advertisement", "documentary"],),
                "language": (["zh", "en", "zh+en"],),
                "max_tokens": ("INT", {"default": 320, "min": 64, "max": 2048, "step": 32}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced",)
    FUNCTION = "execute"
    CATEGORY = "BSAI/LLM"

    def execute(self, engine, raw_prompt, style, language, max_tokens):
        sys = (
            "You are a master visual prompt engineer for video generation. "
            "Rewrite the user's rough idea into a dense, concrete, cinematic prompt. "
            "Add camera, lighting, subject detail, mood, and negative aspects. "
            "Do not explain, output only the rewritten prompt."
        )
        user = f"Style: {style}. Language: {language}.\nRough idea: {raw_prompt}"
        if str(engine).startswith("BSAI Strata"):
            tag = "Strata"
        elif str(engine).startswith("BSAI Nemotron"):
            tag = "Nemotron"
        else:
            tag = "NInfer"
        try:
            if tag == "Strata":
                StrataManager.ensure_running()
                r = StrataManager.chat(
                    [{"role": "system", "content": sys},
                     {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=0.8,
                )
            elif tag == "Nemotron":
                NemotronManager.ensure_running()
                r = NemotronManager.chat(
                    [{"role": "system", "content": sys},
                     {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=0.8,
                )
            else:
                NInferManager.ensure_running()
                r = NInferManager.chat(
                    [{"role": "system", "content": sys},
                     {"role": "user", "content": user}],
                    max_tokens=max_tokens, temperature=0.8,
                )
            text = r["choices"][0]["message"]["content"].strip()
        except Exception as e:
            text = f"[BSAI-LLM-Strata {tag} Enhance ERROR] {type(e).__name__}: {e}"
        return (text,)


# ---------------------------------------------------------------------------
# Web routes
# ---------------------------------------------------------------------------
async def _web_health(request):
    from aiohttp import web
    h = StrataManager.health()
    install = StrataManager.find_install()
    return web.json_response({
        "ok": h.get("ok"),
        "detail": h,
        "install_dir": install,
        "run_scripts": StrataManager.list_run_scripts(),
        "model_dir": STRATA_MODEL_DIR,
        "model_dir_exists": os.path.isdir(STRATA_MODEL_DIR),
        "endpoint": DEFAULT_STRATA_BASE_URL,
        "preferred_quant": StrataManager.get_preferred_quant(),
        "auto_detected_quant": StrataManager.detect_best_quant(),
        "quant_options": list(QUANT_OPTIONS.keys()),
    })


async def _web_start(request):
    from aiohttp import web
    script = request.query.get("script")
    quant = request.query.get("quant")
    if quant and not script:
        script = StrataManager._bat_for_quant(quant)
    r = StrataManager.start(run_script=script)
    return web.json_response(r)


async def _web_set_quant(request):
    from aiohttp import web
    label = request.query.get("quant", "")
    ok = StrataManager.set_preferred_quant(label)
    return web.json_response({"ok": ok, "preferred": StrataManager.get_preferred_quant()})


async def _web_setup(request):
    from aiohttp import web
    quant = request.query.get("quant", "Q2_0")
    context = int(request.query.get("context", "65536"))
    kv = request.query.get("kv", "q4_0")
    r = StrataManager.setup(quant=quant, context=context, kv=kv)
    return web.json_response(r)


async def _web_ninfer_health(request):
    from aiohttp import web
    h = NInferManager.health()
    return web.json_response({
        "ok": h.get("ok"),
        "detail": h,
        "endpoint": NInferManager.base_url(),
        "engine": NInferManager.find_engine(),
        "model": NInferManager.find_model(),
    })


async def _web_ninfer_start(request):
    from aiohttp import web
    ok, msg = NInferManager.ensure_running()
    return web.json_response({"ok": ok, "message": msg, "endpoint": NINFER_BASE_URL})


async def _web_ninfer_models(request):
    from aiohttp import web
    return web.json_response({"models": NInferManager.models()})


async def _web_nemotron_health(request):
    from aiohttp import web
    h = NemotronManager.health()
    return web.json_response({
        "ok": h.get("ok"),
        "detail": h,
        "endpoint": NemotronManager.base_url(),
        "engine": NemotronManager.find_engine(),
        "model": NemotronManager.find_model(),
        "mmproj": NemotronManager.find_mmproj(),
    })


async def _web_nemotron_start(request):
    from aiohttp import web
    ok, msg = NemotronManager.ensure_running()
    return web.json_response({"ok": ok, "message": msg, "endpoint": NEMOTRON_BASE_URL})


async def _web_nemotron_models(request):
    from aiohttp import web
    return web.json_response({"models": NemotronManager.models()})


async def _web_nemotron_stop(request):
    from aiohttp import web
    return web.json_response(NemotronManager.stop())


def setup_web_routes(app):
    app.router.add_get("/bsai/strata/health", _web_health)
    app.router.add_get("/bsai/strata/start", _web_start)
    app.router.add_get("/bsai/strata/set_quant", _web_set_quant)
    app.router.add_get("/bsai/strata/setup", _web_setup)
    app.router.add_get("/bsai/ninfer/health", _web_ninfer_health)
    app.router.add_get("/bsai/ninfer/start", _web_ninfer_start)
    app.router.add_get("/bsai/ninfer/models", _web_ninfer_models)
    app.router.add_get("/bsai/nemotron/health", _web_nemotron_health)
    app.router.add_get("/bsai/nemotron/start", _web_nemotron_start)
    app.router.add_get("/bsai/nemotron/models", _web_nemotron_models)
    app.router.add_get("/bsai/nemotron/stop", _web_nemotron_stop)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "BSAI_LLM_Call": BSAI_LLM_Call,
    "BSAI_LLM_VisionDescribe": BSAI_LLM_VisionDescribe,
    "BSAI_LLM_PromptEnhance": BSAI_LLM_PromptEnhance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BSAI_LLM_Call": "BSAI LLM Call (Strata / Qwen3.8-Flash-Next)",
    "BSAI_LLM_VisionDescribe": "BSAI LLM Vision Describe (Strata mmproj)",
    "BSAI_LLM_PromptEnhance": "BSAI LLM Prompt Enhance (Strata)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "StrataManager",
           "NInferManager", "NemotronManager", "setup_web_routes",
           "DEFAULT_STRATA_BASE_URL", "DEFAULT_STRATA_MODEL", "NINFER_BASE_URL",
           "NINFER_PORT", "NEMOTRON_BASE_URL", "NEMOTRON_PORT"]
