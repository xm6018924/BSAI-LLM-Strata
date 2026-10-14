# BSAI-LLM-Strata

> Local **Qwen3.8-Flash-Next (125B MoE)** inference for ComfyUI, served via [Strata](https://github.com/Niko1221/Strata) and exposed as an OpenAI-compatible HTTP API.
>
> 本地 **Qwen3.8-Flash-Next (125B MoE)** 大模型推理插件，通过 [Strata](https://github.com/Niko1221/Strata) 引擎在本地运行，以 OpenAI 兼容 API 形式供 ComfyUI 全家桶调用。

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

| Component | Spec |
|---|---|
| OS | Windows 10/11 (x64) |
| Python | 3.10 – 3.14 (ComfyUI embedded Python recommended) |
| VRAM | ≥ 16 GB (Q2_0 quant), 24 GB+ recommended |
| RAM | ≥ 32 GB (125B MoE loads ~24 GB at Q2_0) |
| Disk | ~30 GB free for model + runtime |

---

## Installation / 安装步骤

### 1. Install this plugin

Clone into ComfyUI's `custom_nodes/` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-LLM-Strata.git
```

### 2. Auto-download Strata engine (first run)

**No manual clone needed.** The plugin automatically detects if the Strata engine is missing and runs:

```
git clone https://github.com/Niko1221/Strata.git Strata/
```

You'll see this in the ComfyUI console on first use:

```
[BSAI-LLM-Strata] 首次使用，正在自动下载 Strata 引擎到 .../BSAI-LLM-Strata/Strata …
[BSAI-LLM-Strata] Strata 引擎下载完成。
```

> If you already have Strata installed elsewhere (`G:\Strata`, `C:\Strata`, `~/Strata`, etc.), the plugin will find it automatically and skip the download.

### 3. Download model files from network drive

**One unified folder.** Download everything from the network drive and extract into:

```
ComfyUI/models/LLM/Qwen3.8-Flash-Next/
```

Final directory layout:

```
models/LLM/Qwen3.8-Flash-Next/
├── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf   (~14 GB)
├── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf   (~10 GB)
├── mmproj-Qwen3.8-Flash-Next-BF16.gguf                    (~1.5 GB, vision)
├── mtp-q2_0.gguf                                          (~2 GB, spec decoding)
├── rt/                                                    (expert cache, ~1 GB)
└── Strata-data/                                           (pre-converted runtime, ~6 GB)
    ├── packs/q2_0/                                       (packed experts + tokenizer)
    └── mtp/rt/                                           (MTP tensors)
```

Total download: **~24 GB** for text-only, **~26 GB** with vision.

> **You do NOT need to run setup.py.** The `Strata-data/` folder is pre-converted — just extract it alongside the GGUFs.

### 4. One-click setup + start

Open in browser (adjust port to your ComfyUI port):

```
http://127.0.0.1:8188/bsai/strata/setup?quant=Q2_0&context=262144&kv=q4_0
```

Then:

```
http://127.0.0.1:8188/bsai/strata/start
```

Wait until `/v1/models` returns `"ok": true` (model loading takes 1–3 minutes on first run).

### 5. Restart ComfyUI

Nodes appear under `BSAI/LLM` category.

> **After first setup**, you never need to manually start Strata again. Any BSAI node that selects Strata as backend will auto-start it on demand.

---

## Quick Start / 快速开始

After installation, you don't need to manually start Strata anymore:

1. In any BSAI node that supports Strata backend, select **"BSAI Strata"** from the backend dropdown.
2. Execute the node — it will automatically:
   - Probe `http://127.0.0.1:8080/v1/models`
   - If down, launch `run-q2_0.bat` in a detached console
   - Poll health every 3 seconds until the model is ready (max 4 minutes)
   - Send your API request

```
[BSAI-LLM-Strata] 后端未运行，自动启动 Strata…
[BSAI-LLM-Strata] 后端已就绪: ['qwen3.8-flash-next-q2_0']
[BSAI_Qwen_Prompt_Enhancer] Strata 后端就绪: started_and_ready
```

---

## Integration with BSAI Ecosystem / 与 BSAI 全家桶的联动

This plugin is the **central LLM backend** for the entire BSAI ComfyUI plugin suite. Any BSAI node that offers a "Strata" backend option will automatically discover, start, and call this service.

### Nodes with built-in Strata support / 已内置 Strata 选项的节点

| Plugin | Node | How it connects |
|---|---|---|
| **BSAI-Qwen-Prompt-Enhancer** | `BSAI_Qwen_Prompt_Enhancer` | Backend dropdown → `BSAI Strata (Qwen3.8-Flash-Next @127.0.0.1:8080)`; auto-start ensured |
| **BSAI-NPU-Service** | `BSAINPU_LLMGenerate` | Backend combo → Strata; falls back to NPU if Strata unavailable |
| **BSAI-ComfyUI-Sol-H3** | `PromptProvider` | `provider_mode = bsai_strata`; feeds H3 video prompt generation |
| **BSAI-MiniMAX-H3-Prompt** | `H3 ModelLoader` / `Direct Mode` | `model_family = BSAI Strata`; auto-probes port 8080 |
| **BSAI_ComfyUI_Nodes** | `BSAI_QwenModelLoader`, `BSAI_MultiImageVideoReverse` | Backend combo → Strata |

### How auto-discovery works / 自动发现机制

`StrataManager.find_install()` searches (in order):

1. `BSAI-LLM-Strata/Strata/` (bundled, preferred)
2. `G:\Strata`, `C:\Strata`, `D:\Strata`, `E:\Strata`, `F:\Strata`
3. `G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\Strata`
4. `~/Strata`, `~/ProgramMine/Strata`

Any `run-*.bat` found in the install directory becomes a launch candidate. The first one (`run-q2_0.bat` by default) is used for auto-start.

### API contract / API 契约

All BSAI nodes call this endpoint:

```
POST http://127.0.0.1:8080/v1/chat/completions
Authorization: Bearer strata
Content-Type: application/json

{
  "model": "qwen3.8-flash-next-q2_0",
  "messages": [...],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

---

## Web Endpoints / 管理端点

| Endpoint | Description |
|---|---|
| `GET /bsai/strata/health` | Returns `{ok, models, install_dir, run_scripts, endpoint}` |
| `GET /bsai/strata/start` | Spawns first `run-*.bat`; returns `{ok, pid}` |
| `GET /bsai/strata/start?script=run-q4_k_m.bat` | Spawns a specific quant |
| `GET /bsai/strata/setup?quant=Q2_0` | Runs `setup.py` to configure the runtime |

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

---

## License

This plugin code is MIT. The bundled [Strata](https://github.com/Niko1221/Strata) engine and Qwen model weights follow their respective licenses.

---

## Changelog

- **v1.1.0** (2026-10-14): Auto-start on demand (`ensure_running`), WinError 10035 blocking-socket fix, bilingual README
- **v1.0.0**: Initial release — health/start/setup web routes, 3 nodes, OpenAI-compatible proxy
