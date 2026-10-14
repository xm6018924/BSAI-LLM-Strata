# BSAI-LLM-Strata

> Local **Qwen3.8-Flash-Next (125B MoE)** inference for ComfyUI, served via [Strata](https://github.com/Niko1221/Strata) and exposed as an OpenAI-compatible HTTP API.
>
> 本地 **Qwen3.8-Flash-Next (125B MoE)** 大模型推理插件，通过 [Strata](https://github.com/Niko1221/Strata) 引擎在本地运行，以 OpenAI 兼容 API 形式供 ComfyUI 全家桶调用。

---

## What is this? / 这是什么？

**English**: This plugin turns your NVIDIA GPU into a local 125B-parameter LLM server. It wraps the open-source [Strata](https://github.com/Niko1221/Strata) engine (a fast C++ MoE inference runtime) and exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1`. Any ComfyUI node that speaks OpenAI API can call it directly — no cloud, no API key, no per-token cost.

**中文**: 本插件把你的 NVIDIA 显卡变成一个本地 125B 参数的大模型服务器。它封装了开源的 [Strata](https://github.com/Niko1221/Strata) 引擎（一个高速 C++ MoE 推理运行时），在 `http://127.0.0.1:8080/v1` 暴露 OpenAI 兼容 API。任何支持 OpenAI API 的 ComfyUI 节点都可以直接调用——不联网、不填 API Key、不花一分钱。

---

## Features / 功能

- **OpenAI-compatible endpoint** at `http://127.0.0.1:8080/v1` — any ComfyUI node that speaks OpenAI API can point here directly.
- **Auto-start on demand** — any BSAI node that selects Strata as its backend will probe health first; if Strata isn't running, it automatically launches `run-<quant>.bat` and waits (up to 4 minutes) for the 125B model to load before sending the first request.
- **3 built-in ComfyUI nodes**:
  - `BSAI LLM Call` — plain chat completion
  - `BSAI LLM Vision Describe` — image → text (via mmproj vision projector)
  - `BSAI LLM Prompt Enhance` — rough prompt → cinematic prompt
- **Web management endpoints**:
  - `GET /bsai/strata/health` — health check + auto-detect Strata install & run scripts
  - `GET /bsai/strata/start?script=...` — launch a `run-<quant>.bat` (defaults to first found)
  - `GET /bsai/strata/setup?quant=Q2_0&context=262144&kv=q4_0` — one-click `setup.py` against local GGUFs
- **WinError 10035 fix** — uses manually-created blocking sockets, bypassing Python 3.14's asyncio non-blocking default on Windows.

---

## Requirements / 环境要求

| Component | Spec / 要求 |
|---|---|
| OS | Windows 10/11 (x64) |
| GPU | NVIDIA with ≥ 16 GB VRAM (Q2_0 quant); 24 GB+ recommended |
| RAM | ≥ 32 GB (125B MoE loads ~24 GB at Q2_0) |
| Disk | ~30 GB free for model + runtime files |
| ComfyUI | Portable/embedded Python build recommended (Python 3.10 – 3.14) |

> **Note / 注意**: This uses CUDA. AMD / Intel Arc / CPU-only setups are not supported by the Strata engine itself.

---

## Installation / 安装步骤

### Step 1: Install this plugin / 安装插件

Clone into ComfyUI's `custom_nodes/` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-LLM-Strata.git
```

Or download the ZIP and extract it into `custom_nodes/BSAI-LLM-Strata/`.

---

### Step 2: Auto-download Strata engine / 自动下载 Strata 引擎

**No manual clone needed.** On first use, the plugin automatically detects if the Strata engine is missing and runs:

```
git clone https://github.com/Niko1221/Strata.git Strata/
```

You'll see this in the ComfyUI console:

```
[BSAI-LLM-Strata] 首次使用，正在自动下载 Strata 引擎到 .../BSAI-LLM-Strata/Strata …
[BSAI-LLM-Strata] Strata 引擎下载完成。
```

> If you already have Strata installed elsewhere (`G:\Strata`, `C:\Strata`, `~/Strata`, etc.), the plugin will find it automatically and skip the download.

---

### Step 3: Download model files / 下载模型文件

**One unified folder.** Download all model files from the network drive (百度网盘 / 阿里云盘 / 夸克网盘 — link provided in the release page), then extract into:

```
ComfyUI/models/LLM/Qwen3.8-Flash-Next/
```

Final directory layout after extraction:

```
ComfyUI/models/LLM/Qwen3.8-Flash-Next/
├── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf   (~14 GB)
├── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf   (~10 GB)
├── mmproj-Qwen3.8-Flash-Next-BF16.gguf                    (~1.5 GB, vision)
├── mtp-q2_0.gguf                                          (~2 GB, spec decoding)
├── rt/                                                    (expert cache, ~1 GB)
└── Strata-data/                                           (pre-converted runtime, ~6 GB)
    ├── packs/q2_0/                                       (packed experts + tokenizer)
    └── mtp/rt/                                           (MTP tensors)
```

Total download: **~26 GB** (includes vision support).

> **You do NOT need to run setup.py.** The `Strata-data/` folder is pre-converted — just extract it alongside the GGUFs. This saves you 30+ minutes of conversion time.

---

### Step 4: Start Strata / 启动 Strata

**Option A: Automatic (recommended)** — Just restart ComfyUI. When you first run a node that uses Strata backend, it will auto-start itself.

**Option B: Manual** — Open in browser (adjust port to your ComfyUI port):

```
http://127.0.0.1:8188/bsai/strata/start
```

Or double-click `custom_nodes/BSAI-LLM-Strata/Strata/run-q2_0.bat`.

Wait until the model finishes loading (1–3 minutes on first run). You'll see `Strata listening on http://127.0.0.1:8080` in the console.

---

### Step 5: Restart ComfyUI / 重启 ComfyUI

Nodes appear under the `BSAI/LLM` category in the node browser.

---

## Quick Start / 快速开始

After installation, you don't need to manually start Strata anymore:

1. In any BSAI node that supports Strata backend, select **"BSAI Strata"** from the backend dropdown.
2. Execute the node — it will automatically:
   - Probe `http://127.0.0.1:8080/v1/models`
   - If down, launch `run-q2_0.bat` in a detached console
   - Poll health every 3 seconds until the model is ready (max 4 minutes)
   - Send your API request

Console output looks like:

```
[BSAI-LLM-Strata] 后端未运行，自动启动 Strata…
[BSAI-LLM-Strata] 后端已就绪: ['qwen3.8-flash-next-q2_0']
[BSAI_Qwen_Prompt_Enhancer] Strata 后端就绪: started_and_ready
```

---

## Built-in Nodes / 内置节点

| Node | What it does |
|---|---|
| **BSAI LLM Call** | Plain chat completion — send a prompt, get a text response. Use for any LLM task. |
| **BSAI LLM Vision Describe** | Image → text description. Upload an image, get a detailed caption (uses mmproj vision projector). |
| **BSAI LLM Prompt Enhance** | Rough prompt → cinematic prompt. Type "a girl walking in the rain", get a full cinematic prompt ready for H3 / Wan / SVD video generation. |

---

## Integration with BSAI Ecosystem / 与 BSAI 全家桶的联动

This plugin is the **central LLM backend** for the entire BSAI ComfyUI plugin suite. Any BSAI node that offers a "Strata" backend option will automatically discover, start, and call this service.

### Nodes with built-in Strata support / 已内置 Strata 选项的节点

| Plugin / 插件 | Node / 节点 | How to select / 如何选择 |
|---|---|---|
| **BSAI-Qwen-Prompt-Enhancer** | `BSAI_Qwen_Prompt_Enhancer` | Backend dropdown → `BSAI Strata` |
| **BSAI-NPU-Service** | `BSAINPU_LLMGenerate` | Backend combo → `BSAI Strata HTTP` |
| **BSAI-ComfyUI-Sol-H3** | `PromptProvider` | `provider_mode = bsai_strata` |
| **BSAI-MiniMAX-H3-Prompt** | `H3 ModelLoader` | `model_family = BSAI Strata` |
| **BSAI_ComfyUI_Nodes** | `BSAI_QwenModelLoader`, `BSAI_MultiImageVideoReverse` | Backend combo → `BSAI Strata HTTP` |

### How auto-discovery works / 自动发现机制

`StrataManager.find_install()` searches (in order):

1. `BSAI-LLM-Strata/Strata/` (bundled, preferred)
2. `G:\Strata`, `C:\Strata`, `D:\Strata`, `E:\Strata`, `F:\Strata`
3. `G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\Strata`
4. `~/Strata`, `~/ProgramMine/Strata`

Any `run-*.bat` found in the install directory becomes a launch candidate. The first one (`run-q2_0.bat` by default) is used for auto-start.

---

## API Contract / API 契约

All BSAI nodes call this endpoint. You can also call it from any OpenAI-compatible client (ChatBox, NextChat, etc.):

```
POST http://127.0.0.1:8080/v1/chat/completions
Authorization: Bearer strata
Content-Type: application/json

{
  "model": "qwen3.8-flash-next-q2_0",
  "messages": [
    {"role": "user", "content": "Hello, introduce yourself."}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

Model name: `qwen3.8-flash-next-q2_0`. API key can be anything (we use `strata` by default).

---

## Web Endpoints / 管理端点

When ComfyUI is running, these endpoints are available:

| Endpoint | Description / 说明 |
|---|---|
| `GET /bsai/strata/health` | Health check: `{ok, models, install_dir, run_scripts, endpoint}` |
| `GET /bsai/strata/start` | Spawn first `run-*.bat`; returns `{ok, pid}` |
| `GET /bsai/strata/start?script=run-q4_k_m.bat` | Spawn a specific quant |
| `GET /bsai/strata/setup?quant=Q2_0` | Run `setup.py` to configure the runtime |

---

## Troubleshooting / 常见问题

### WinError 10035 on Python 3.14 Windows

**Symptom**: `<urlopen error [WinError 10035] 无法立即完成一个非阻止性套接字操作>`

**Cause**: Python 3.14's asyncio event loop sets sockets to non-blocking by default; `urllib` doesn't handle this for local TCP connect.

**Fix (already applied)**: This plugin uses manually-created blocking sockets (`socket.setblocking(True)`) instead of `urllib.request.urlopen`. If you see this error in a **third-party** plugin calling Strata, ensure it uses the same pattern or goes through `StrataManager.ensure_running()`.

### Strata takes > 4 minutes to load

The auto-start wait timeout is 240 seconds. On slower systems, you may need to:
1. Start Strata manually via `run-q2_0.bat`
2. Or increase `wait_timeout` in `strata_manager.py` → `StrataManager.ensure_running()`

### VRAM OOM

Use a smaller quant: `Q2_0` requires ~24 GB VRAM. If you have 16 GB, switch to `IQ2_XS` or run Strata on CPU only.

### "Strata engine not found"

The auto-download needs `git` installed. If `git clone` fails (e.g. GitHub blocked), manually download the Strata ZIP from [https://github.com/Niko1221/Strata](https://github.com/Niko1221/Strata) and extract it into `custom_nodes/BSAI-LLM-Strata/Strata/`.

### Model files not found

Make sure the GGUF files are in `ComfyUI/models/LLM/Qwen3.8-Flash-Next/` and the `Strata-data/` folder is in the same directory. The plugin auto-detects this path.

---

## License

This plugin code is MIT. The bundled [Strata](https://github.com/Niko1221/Strata) engine and Qwen model weights follow their respective licenses.

---

## Changelog

### v1.3.0 (2026-10-14) — Quant selector + Auto hardware detection
- **新增量化选择下拉菜单** / Added quant dropdown in BSAI LLM Call node:
  - `Auto (自动适配)` — detects VRAM & RAM via nvidia-smi + psutil, picks best quant
  - `Q2_0` — fastest, lowest VRAM (~24 GB)
  - `IQ2_XS` — slightly better quality, similar speed
  - `IQ3_XXS` — better quality, slower
  - `IQ3_S` — best quality (near BF16), needs ≥24 GB VRAM + ≥64 GB RAM
- **Auto-select tiers**: ≥24GB VRAM+64GB RAM → IQ3_S; ≥20GB+48GB → IQ3_XXS; ≥16GB+40GB → IQ2_XS; else → Q2_0
- **Web endpoint**: `GET /bsai/strata/set_quant?quant=<label>` to set global preferred quant
- **Built 4 quant configs + run scripts**: `strata-q2_0.json`, `strata-iq2_xs.json`, `strata-iq3_xxs.json`, `strata-iq3_s.json` + matching `run-*.bat`
- **Converted 3 extra packs** via `tools/iq_pack.py`: `packs/iq2_xs`, `packs/iq3_xxs`, `packs/iq3_s` (abliterated GGUFs)
- **Restored BSAI Strata backend** in BSAI-Qwen-Prompt-Enhancer (was overwritten by upstream update)
- **Cleanup**: deleted `.done` markers + duplicate `Q2_0/` subdir (freed 124.5 GB)

### v1.2.0 (2026-10-14) — Unified model directory
- Strata-data moved into `models/LLM/Qwen3.8-Flash-Next/Strata-data/` (single download)
- All BSAI plugins patched with blocking-socket HTTP (WinError 10035 fix)
- Comprehensive bilingual README

### v1.1.0
- Auto-start on demand (`ensure_running`), WinError 10035 blocking-socket fix

### v1.0.0
- Initial release — health/start/setup web routes, 3 nodes, OpenAI-compatible proxy
