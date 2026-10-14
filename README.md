# BSAI-LLM-Strata

> Local **Qwen3.8-Flash-Next (125B MoE)** inference for ComfyUI, served via [Strata](https://github.com/Niko1221/Strata) and exposed as an OpenAI-compatible HTTP API.
>
> 鏈湴 **Qwen3.8-Flash-Next (125B MoE)** 澶фā鍨嬫帹鐞嗘彃浠讹紝閫氳繃 [Strata](https://github.com/Niko1221/Strata) 寮曟搸鍦ㄦ湰鍦拌繍琛岋紝浠?OpenAI 鍏煎 API 褰㈠紡渚?ComfyUI 鍏ㄥ妗惰皟鐢ㄣ€?
---

## Features / 鍔熻兘

- **OpenAI-compatible endpoint** at `http://127.0.0.1:8080/v1` 鈥?any ComfyUI node that speaks OpenAI API can point here directly.
- **Auto-start on demand** 鈥?any BSAI node that selects Strata as its backend will probe health first; if Strata isn't running, it automatically launches `run-<quant>.bat` and waits (up to 4 minutes) for the 125B model to load before sending the first request.
- **3 built-in ComfyUI nodes**:
  - `BSAI LLM Call` 鈥?plain chat completion
  - `BSAI LLM Vision Describe` 鈥?image 鈫?text (via mmproj vision projector)
  - `BSAI LLM Prompt Enhance` 鈥?rough prompt 鈫?cinematic prompt
- **Web management endpoints**:
  - `GET /bsai/strata/health` 鈥?health check + auto-detect Strata install & run scripts
  - `GET /bsai/strata/start?script=...` 鈥?launch a `run-<quant>.bat` (defaults to first found)
  - `GET /bsai/strata/setup?quant=Q2_0&context=262144&kv=q4_0` 鈥?one-click `setup.py` against local GGUFs
- **WinError 10035 fix** 鈥?uses manually-created blocking sockets, bypassing Python 3.14's asyncio non-blocking default on Windows.

---

## Requirements / 鐜瑕佹眰

| Component | Spec |
|---|---|
| OS | Windows 10/11 (x64) |
| Python | 3.10 鈥?3.14 (ComfyUI embedded Python recommended) |
| VRAM | 鈮?16 GB (Q2_0 quant), 24 GB+ recommended |
| RAM | 鈮?32 GB (125B MoE loads ~24 GB at Q2_0) |
| Disk | ~30 GB free for model + runtime |

---

## Installation / 瀹夎姝ラ

### 1. Clone Strata engine

```bash
cd ComfyUI/custom_nodes/BSAI-LLM-Strata
git clone https://github.com/Niko1221/Strata.git Strata
```

> This bundled `Strata/` subfolder is the **first search location**. The plugin also auto-discovers `G:\Strata`, `C:\Strata`, `D:\Strata`, `~/Strata`, etc.

### 2. Download model GGUFs

Place quantized GGUF files under:

```
ComfyUI/models/LLM/Qwen3.8-Flash-Next/
```

Recommended quants: `Q2_0` (24 GB, lowest VRAM), `IQ3_S`, `Q4_K_M`.

### 3. Run first-time setup

Open in browser (adjust port to your ComfyUI port):

```
http://127.0.0.1:8188/bsai/strata/setup?quant=Q2_0&context=262144&kv=q4_0
```

This runs `setup.py` to wire the model directory to Strata's runtime format.

### 4. Start Strata server

```
http://127.0.0.1:8188/bsai/strata/start
```

Wait until `/v1/models` returns `"ok": true` (model loading takes 1鈥? minutes on first run).

### 5. Restart ComfyUI

Nodes appear under `BSAI/LLM` category.

---

## Quick Start / 蹇€熷紑濮?
After installation, you don't need to manually start Strata anymore:

1. In any BSAI node that supports Strata backend, select **"BSAI Strata"** from the backend dropdown.
2. Execute the node 鈥?it will automatically:
   - Probe `http://127.0.0.1:8080/v1/models`
   - If down, launch `run-q2_0.bat` in a detached console
   - Poll health every 3 seconds until the model is ready (max 4 minutes)
   - Send your API request

```
[BSAI-LLM-Strata] 鍚庣鏈繍琛岋紝鑷姩鍚姩 Strata鈥?[BSAI-LLM-Strata] 鍚庣宸插氨缁? ['qwen3.8-flash-next-q2_0']
[BSAI_Qwen_Prompt_Enhancer] Strata 鍚庣灏辩华: started_and_ready
```

---

## Integration with BSAI Ecosystem / 涓?BSAI 鍏ㄥ妗剁殑鑱斿姩

This plugin is the **central LLM backend** for the entire BSAI ComfyUI plugin suite. Any BSAI node that offers a "Strata" backend option will automatically discover, start, and call this service.

### Nodes with built-in Strata support / 宸插唴缃?Strata 閫夐」鐨勮妭鐐?
| Plugin | Node | How it connects |
|---|---|---|
| **BSAI-Qwen-Prompt-Enhancer** | `BSAI_Qwen_Prompt_Enhancer` | Backend dropdown 鈫?`BSAI Strata (Qwen3.8-Flash-Next @127.0.0.1:8080)`; auto-start ensured |
| **BSAI-NPU-Service** | `BSAINPU_LLMGenerate` | Backend combo 鈫?Strata; falls back to NPU if Strata unavailable |
| **BSAI-ComfyUI-Sol-H3** | `PromptProvider` | `provider_mode = bsai_strata`; feeds H3 video prompt generation |
| **BSAI-MiniMAX-H3-Prompt** | `H3 ModelLoader` / `Direct Mode` | `model_family = BSAI Strata`; auto-probes port 8080 |
| **BSAI_ComfyUI_Nodes** | `BSAI_QwenModelLoader`, `BSAI_MultiImageVideoReverse` | Backend combo 鈫?Strata |

### How auto-discovery works / 鑷姩鍙戠幇鏈哄埗

`StrataManager.find_install()` searches (in order):

1. `BSAI-LLM-Strata/Strata/` (bundled, preferred)
2. `G:\Strata`, `C:\Strata`, `D:\Strata`, `E:\Strata`, `F:\Strata`
3. `G:\BSAI-ComfyUI-intel-XPU-GPU-NPU-aki\Strata`
4. `~/Strata`, `~/ProgramMine/Strata`

Any `run-*.bat` found in the install directory becomes a launch candidate. The first one (`run-q2_0.bat` by default) is used for auto-start.

### API contract / API 濂戠害

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

## Web Endpoints / 绠＄悊绔偣

| Endpoint | Description |
|---|---|
| `GET /bsai/strata/health` | Returns `{ok, models, install_dir, run_scripts, endpoint}` |
| `GET /bsai/strata/start` | Spawns first `run-*.bat`; returns `{ok, pid}` |
| `GET /bsai/strata/start?script=run-q4_k_m.bat` | Spawns a specific quant |
| `GET /bsai/strata/setup?quant=Q2_0` | Runs `setup.py` to configure the runtime |

---

## Troubleshooting / 甯歌闂

### WinError 10035 on Python 3.14 Windows

**Symptom**: `<urlopen error [WinError 10035] 鏃犳硶绔嬪嵆瀹屾垚涓€涓潪闃绘鎬у鎺ュ瓧鎿嶄綔>`

**Cause**: Python 3.14's asyncio event loop sets sockets to non-blocking by default; `urllib` doesn't handle this for local TCP connect.

**Fix (already applied)**: This plugin uses manually-created blocking sockets (`socket.setblocking(True)`) instead of `urllib.request.urlopen`. If you see this error in a **third-party** plugin calling Strata, ensure it uses the same pattern or goes through `StrataManager.ensure_running()`.

### Strata takes > 4 minutes to load

The auto-start wait timeout is 240 seconds. On slower systems, you may need to:
1. Start Strata manually via `run-q2_0.bat`
2. Or increase `wait_timeout` in `strata_manager.py` 鈫?`StrataManager.ensure_running()`

### VRAM OOM

Use a smaller quant: `Q2_0` requires ~24 GB VRAM. If you have 16 GB, switch to `IQ2_XS` or run Strata on CPU only.

---

## License

This plugin code is MIT. The bundled [Strata](https://github.com/Niko1221/Strata) engine and Qwen model weights follow their respective licenses.

---

## Changelog

- **v1.1.0** (2026-10-14): Auto-start on demand (`ensure_running`), WinError 10035 blocking-socket fix, bilingual README
- **v1.0.0**: Initial release 鈥?health/start/setup web routes, 3 nodes, OpenAI-compatible proxy
