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
            ("serve-server.py", os.path.join("serve", "server.py")),
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
        # Sort by speed: Q2_0 fastest, IQ3_S slowest
        order = {"Q2_0": 0, "IQ2_XS": 1, "IQ3_XXS": 2, "IQ3_S": 3, "Q4_XS": 4}
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
                        json_name = f"strata-{best_quant.lower()}.json"
                        bat_content = f'''@echo off
title Strata {best_quant}
cd /d "{install}"
"{venv_python}" "{os.path.join(install, "serve", "server.py")}" "--engine" "strata" "--config" "{os.path.join(install, json_name)}" "--port" "8080" "--open"
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

    # ---------- ensure running (auto-start + wait) ----------
    @classmethod
    def ensure_running(cls, wait_timeout=240.0, poll_interval=3.0, quant_label=None):
        """Block until Strata is reachable. If not running, spawn the run-*.bat
        for the given quant_label (or the user's preferred quant).

        Returns (ok, message). Called by any BSAI node before hitting /v1.
        """
        h = cls.health(timeout=1.0)
        if h.get("ok"):
            return True, "already_running"

        # Not up — spawn it
        label = quant_label or cls.get_preferred_quant()
        if label == "Auto (自动适配)":
            label = cls.detect_best_quant()
        # Auto-generate json config with correct paths (fixes hardcoded paths on new machines)
        ok_gen, msg_gen = cls._refresh_json_config(label)
        if not ok_gen:
            print(f"[BSAI-LLM-Strata] Warning: json config refresh failed: {msg_gen}")
        bat = cls._bat_for_quant(label)
        print(f"[BSAI-LLM-Strata] 后端未运行，自动启动 Strata ({label})…")
        start_r = cls.start(run_script=bat)
        if not start_r.get("ok"):
            return False, f"启动失败: {start_r.get('error')}"
        if start_r.get("already_running"):
            # health was racy; re-probe
            pass

        # Wait for model to load (125B MoE q2_0 takes 60~180s)
        deadline = time.time() + wait_timeout
        last_err = "?"
        while time.time() < deadline:
            time.sleep(poll_interval)
            h = cls.health(timeout=2.0)
            if h.get("ok"):
                print(f"[BSAI-LLM-Strata] 后端已就绪: {h.get('models')}")
                return True, "started_and_ready"
            last_err = h.get("error", "?")
        return False, f"等待超时({wait_timeout:.0f}s): {last_err}"

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
        ggufs = sorted([f for f in os.listdir(gguf_dir) if f.endswith(".gguf")])
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
        max_context_val = _get_arg("--max-context", "32768")

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
        ]

        new_cfg = {
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
    def chat(messages, model=DEFAULT_STRATA_MODEL, max_tokens=1024,
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
# ComfyUI nodes
# ---------------------------------------------------------------------------
class BSAI_LLM_Call:
    """One-shot chat completion via local Strata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                      "tooltip": "User prompt / 用户提示词"}),
                "system": ("STRING", {"multiline": True, "default": "You are a helpful assistant.",
                                     "tooltip": "System prompt / 系统提示词"}),
                "quant": (list(QUANT_OPTIONS.keys()), {"default": DEFAULT_QUANT_LABEL,
                           "tooltip": "Which Strata quant to auto-start (if not running)"}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 32768, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "model": ("STRING", {"default": DEFAULT_STRATA_MODEL,
                                    "tooltip": "Model name served by Strata"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "execute"
    CATEGORY = "BSAI/LLM"
    DESCRIPTION = "Call local Qwen3.8-Flash-Next via Strata (OpenAI-compatible)"

    def execute(self, prompt, system, quant, max_tokens, temperature, model=DEFAULT_STRATA_MODEL):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            ok, msg = StrataManager.ensure_running(quant_label=quant)
            if not ok:
                return (f"[BSAI-LLM-Strata] Strata 启动失败: {msg}",)
            r = StrataManager.chat(msgs, model=model, max_tokens=max_tokens,
                                   temperature=temperature)
            text = r["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"[BSAI-LLM-Strata ERROR] {type(e).__name__}: {e}\n" \
                   f"Is Strata running at http://127.0.0.1:8080? Check /bsai/strata/health."
        return (text,)


class BSAI_LLM_VisionDescribe:
    """Image -> text description. Sends base64 data URL to Strata's mmproj."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
    DESCRIPTION = "Vision: ask Qwen3.8-Flash-Next about an input image (uses mmproj)"

    def execute(self, image, prompt, max_tokens):
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
        try:
            r = StrataManager.chat(msgs, max_tokens=max_tokens, temperature=0.2)
            text = r["choices"][0]["message"]["content"]
        except Exception as e:
            text = f"[BSAI-LLM-Strata Vision ERROR] {type(e).__name__}: {e}"
        return (text,)


class BSAI_LLM_PromptEnhance:
    """Wrap a rough prompt into a richer cinematic prompt via local LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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

    def execute(self, raw_prompt, style, language, max_tokens):
        sys = (
            "You are a master visual prompt engineer for video generation. "
            "Rewrite the user's rough idea into a dense, concrete, cinematic prompt. "
            "Add camera, lighting, subject detail, mood, and negative aspects. "
            "Do not explain, output only the rewritten prompt."
        )
        user = f"Style: {style}. Language: {language}.\nRough idea: {raw_prompt}"
        try:
            r = StrataManager.chat(
                [{"role": "system", "content": sys},
                 {"role": "user", "content": user}],
                max_tokens=max_tokens, temperature=0.8,
            )
            text = r["choices"][0]["message"]["content"].strip()
        except Exception as e:
            text = f"[BSAI-LLM-Strata Enhance ERROR] {type(e).__name__}: {e}"
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


def setup_web_routes(app):
    app.router.add_get("/bsai/strata/health", _web_health)
    app.router.add_get("/bsai/strata/start", _web_start)
    app.router.add_get("/bsai/strata/set_quant", _web_set_quant)
    app.router.add_get("/bsai/strata/setup", _web_setup)


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
           "setup_web_routes", "DEFAULT_STRATA_BASE_URL", "DEFAULT_STRATA_MODEL"]
