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

- **One-click install**: Download the full model package from the network drive, extract it, install the plugin — everything else is automatic.
- **Engine auto-copy**: The plugin automatically detects the Strata engine binary in your model directory and copies it to the right place.
- **OpenAI-compatible endpoint** at `http://127.0.0.1:8080/v1` — any ComfyUI node that speaks OpenAI API can point here directly.
- **Auto-start on demand** — any BSAI node that selects Strata as its backend will probe health first; if Strata isn't running, it automatically launches `run-<quant>.bat` and waits (up to 4 minutes) for the 125B model to load before sending the first request.
- **Auto-generated config paths** — no more hardcoded `G:\...` paths. On startup, the plugin scans your actual model directory, finds the GGUF files, mmproj, packs, and MTP tensors, and regenerates `strata-<quant>.json` with correct absolute paths for your machine.
- **3 built-in ComfyUI nodes**:
  - `BSAI LLM Call` — plain chat completion
  - `BSAI LLM Vision Describe` — image → text (via mmproj vision projector)
  - `BSAI LLM Prompt Enhance` — rough prompt → cinematic prompt

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

### Step 1: Download the full model package / 下载完整模型包

**English**: Go to the download page and get the full model package from Quark Pan (夸克网盘):

**Download link / 下载链接**: https://pan.quark.cn/s/f6d2bbdaaa18

The package is **all-in-one**: it includes the model weights, pre-converted runtime data, AND the Strata engine binary — no separate downloads needed.

**中文**: 前往夸克网盘下载完整的模型包：

**下载链接**: https://pan.quark.cn/s/f6d2bbdaaa18

这个包是**一体化的**：包含模型权重、预转换的运行时数据，**以及 Strata 引擎二进制文件**——不需要额外下载。

---

### Step 2: Extract the model package / 解压模型包

**English**: Extract the downloaded ZIP into your ComfyUI models directory:

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
├── engine/                                                (Strata engine binary, ~200 MB)
│   ├── strata.exe
│   ├── strata-vision.exe
│   └── BUILD.json
├── rt/                                                    (expert cache, ~1 GB)
└── Strata-data/                                           (pre-converted runtime, ~6 GB)
    ├── packs/q2_0/                                       (packed experts + tokenizer)
    └── mtp/rt/                                           (MTP tensors)
```

Total download: **~26 GB** (includes vision support + engine binary).

**中文**: 把下载的 ZIP 解压到 ComfyUI 的模型目录：

```
ComfyUI/models/LLM/Qwen3.8-Flash-Next/
```

解压后的目录结构：

```
ComfyUI/models/LLM/Qwen3.8-Flash-Next/
├── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf   (~14 GB)
├── Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf   (~10 GB)
├── mmproj-Qwen3.8-Flash-Next-BF16.gguf                    (~1.5 GB, 视觉模型)
├── mtp-q2_0.gguf                                          (~2 GB, 投机解码)
├── rt/                                                    (专家缓存, ~1 GB)
└── Strata-data/                                           (预转换运行时, ~6 GB)
    ├── engine/                                            (Strata 引擎二进制, ~200 MB)
    │   ├── strata.exe
    │   ├── strata-vision.exe
    │   └── BUILD.json
    ├── packs/q2_0/                                       (打包专家 + 分词器)
    └── mtp/rt/                                           (MTP 张量)
```

总下载量：**约 26 GB**（包含视觉支持 + 引擎二进制）。

> **You do NOT need to run setup.py.** The `Strata-data/` folder is pre-converted and includes the engine — just extract it alongside the GGUFs. This saves you 30+ minutes of conversion and download time.
>
> **你不需要运行 setup.py。** `Strata-data/` 目录已经预转换好，并且包含引擎文件——只要把它和 GGUF 文件放在一起就行。这能帮你节省 30 分钟以上的转换和下载时间。

---

### Step 3: Install the plugin / 安装插件

**English**: Install this plugin via ComfyUI Manager, or clone the repo into `custom_nodes/`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-LLM-Strata.git
```

**中文**: 通过 ComfyUI Manager 安装本插件，或者直接 clone 到 `custom_nodes/` 目录：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xm6018924/BSAI-LLM-Strata.git
```

> **Note / 注意**: You do NOT need `--recursive`. The engine is already in your model package.
>
> **注意**: 你不需要加 `--recursive` 参数。引擎已经在你的模型包里了。

---

### Step 4: Restart ComfyUI / 重启 ComfyUI

**English**: That's it! Just restart ComfyUI. The plugin will automatically:

1. **Detect the model directory** at `models/LLM/Qwen3.8-Flash-Next/`
2. **Copy the engine binary** from `Strata-data/engine/` to the plugin's `Strata/engine/` directory
3. **Auto-detect the available quant** (Q2_0, IQ2_XS, etc.)
4. **Generate the config file** (`strata-<quant>.json`) with correct absolute paths
5. **Generate the launch script** (`run-<quant>.bat`)

You'll see this in the console:

```
[BSAI-LLM-Strata] Strata engine found: .../Strata
[BSAI-LLM-Strata] 使用系统 Python: C:\...\Python312\python.exe
[BSAI-LLM-Strata] 检测到模型目录下的引擎文件: .../engine
[BSAI-LLM-Strata] 引擎文件已复制到插件目录: .../Strata/engine
[BSAI-LLM-Strata] 检测到已有的量化模型: Q2_0
[BSAI-LLM-Strata] 配置文件生成完成。
[BSAI-LLM-Strata] 启动脚本已生成: run-q2_0.bat
[BSAI-LLM-Strata] Strata engine fully installed. Ready to use.
```

**中文**: 就这么简单！重启 ComfyUI 就行。插件会自动：

1. **检测模型目录** `models/LLM/Qwen3.8-Flash-Next/`
2. **从 `Strata-data/engine/` 复制引擎二进制**到插件的 `Strata/engine/` 目录
3. **自动检测可用的量化版本**（Q2_0、IQ2_XS 等）
4. **生成配置文件**（`strata-<quant>.json`），所有路径自动适配
5. **生成启动脚本**（`run-<quant>.bat`）

你会在控制台看到：

```
[BSAI-LLM-Strata] Strata engine found: .../Strata
[BSAI-LLM-Strata] 使用系统 Python: C:\...\Python312\python.exe
[BSAI-LLM-Strata] 检测到模型目录下的引擎文件: .../engine
[BSAI-LLM-Strata] 引擎文件已复制到插件目录: .../Strata/engine
[BSAI-LLM-Strata] 检测到已有的量化模型: Q2_0
[BSAI-LLM-Strata] 配置文件生成完成。
[BSAI-LLM-Strata] 启动脚本已生成: run-q2_0.bat
[BSAI-LLM-Strata] Strata engine fully installed. Ready to use.
```

> **Total first-time setup: ~30 seconds** (just copying the engine file + generating configs).
>
> **首次安装总耗时：约 30 秒**（只是复制引擎文件 + 生成配置）。

---

### Step 5: Start using Strata / 开始使用 Strata

**English**: There's no need to manually start Strata. Just use any BSAI node that supports Strata backend:

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

**中文**: 你不需要手动启动 Strata。只要使用任何支持 Strata 后端的 BSAI 节点就行：

1. 在任何支持 Strata 后端的 BSAI 节点里，从后端下拉菜单选择 **"BSAI Strata"**。
2. 执行节点——它会自动：
   - 探测 `http://127.0.0.1:8080/v1/models`
   - 如果没运行，在单独的控制台窗口启动 `run-q2_0.bat`
   - 每 3 秒检查一次健康状态，直到模型就绪（最多等 4 分钟）
   - 发送 API 请求

控制台输出示例：

```
[BSAI-LLM-Strata] 后端未运行，自动启动 Strata…
[BSAI-LLM-Strata] 后端已就绪: ['qwen3.8-flash-next-q2_0']
[BSAI_Qwen_Prompt_Enhancer] Strata 后端就绪: started_and_ready
```

---

## Quick Start / 快速开始

After installation, you don't need to manually start Strata anymore. Just select "BSAI Strata" as the backend in any BSAI node, and it will handle the rest.

安装完成后，你不需要手动启动 Strata。只要在任何 BSAI 节点里选择 "BSAI Strata" 作为后端，剩下的它都会自动处理。

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
| `GET /bsai/strata/start?script=run-q2_0.bat` | Spawn a specific quant |

---

## Troubleshooting / 常见问题

### "Model directory not found"

Make sure you extracted the model package into `ComfyUI/models/LLM/Qwen3.8-Flash-Next/`. The plugin looks for this exact path.

### "Engine binary not found"

Make sure the `engine/` folder exists in your model directory (`models/LLM/Qwen3.8-Flash-Next/engine/`). It should contain `strata.exe` and `strata-vision.exe`. If not, re-download the full model package from the network drive.

### ComfyUI window closes automatically after starting / ComfyUI 窗口启动后自动关闭

**Symptom / 症状**: After ComfyUI starts, the BSAI launcher window closes automatically, but a separate "Strata" window appears in the background.

**原因**: Strata 服务启动时会加载约 40 GB 模型到内存，占用大量系统资源。如果你的内存刚好不够（比如 64 GB 内存同时运行 ComfyUI + Strata），ComfyUI 可能会因为内存不足而崩溃退出。

**Solution / 解决方法**:

1. **Don't worry — Strata is still running.** Look for the separate "Strata" console window in your taskbar. It's loading the model (1-3 minutes on first run).

   **别慌——Strata 服务还在后台运行。** 在任务栏找到单独的 "Strata" 控制台窗口，它正在加载模型（首次需要 1-3 分钟）。

2. **Wait for Strata to finish loading.** You'll see `Strata listening on http://127.0.0.1:8080` in that window.

   **等 Strata 加载完成。** 窗口里会显示 `Strata listening on http://127.0.0.1:8080`。

3. **Then restart ComfyUI normally.** Now that Strata is already running (and loaded), ComfyUI won't need to start it again, and there won't be a memory spike.

   **然后正常重启 ComfyUI。** 现在 Strata 已经在运行（模型已加载），ComfyUI 启动时不会再触发内存峰值，就不会崩溃了。

### VRAM OOM

Use a smaller quant: `Q2_0` requires ~24 GB VRAM. If you have 16 GB, switch to `IQ2_XS` or run Strata on CPU only.

---

## License

This plugin code is MIT. The bundled [Strata](https://github.com/Niko1221/Strata) engine and Qwen model weights follow their respective licenses.

---

## Changelog

### v2.0.0 (2026-10-15) — All-in-one model package + Zero-config install
- **English**: Complete redesign of the installation flow.
  - The full model package (weights + runtime data + engine binary) is downloaded as one ZIP from the network drive.
  - The plugin automatically detects the engine in `Strata-data/engine/` and copies it to the plugin directory.
  - No more `setup.py`, no more venv, no more downloading from GitHub on first run.
  - **First-time setup takes ~30 seconds** (just copying files + generating configs).
- **中文**: 完全重新设计了安装流程。
  - 完整模型包（权重 + 运行时数据 + 引擎二进制）作为一个 ZIP 从网盘下载。
  - 插件自动检测 `Strata-data/engine/` 里的引擎文件，复制到插件目录。
  - 不再需要 `setup.py`、不再需要 venv、首次启动时不再需要从 GitHub 下载。
  - **首次安装只需约 30 秒**（只是复制文件 + 生成配置）。
- **Removed venv requirement**: The plugin now uses system Python directly, no virtual environment needed.
- **Simplified dependencies**: Removed cmake, ninja and other build-time dependencies — the engine is pre-built.
- **Automatic engine copy**: The plugin auto-detects the engine in the model directory's `Strata-data/engine/` and copies it to the plugin's `Strata/engine/` directory.

### v1.5.0 (2026-10-15) — Strata engine as git submodule + Auto-download
- Strata engine bundled as git submodule, auto-download on first ComfyUI startup.
- Background auto-download thread, no blocking ComfyUI startup.

### v1.4.0 (2026-10-15) — Auto-path config generation + Speed tuning
- Auto-scan model directory, regenerate `strata-<quant>.json` with correct absolute paths.
- Speed tuning: context size, spec decoding, KV cache, VRAM reserve optimized.

### v1.3.0 (2026-10-14) — Quant selector + Auto hardware detection
- Added quant dropdown: Auto, Q2_0, IQ2_XS, IQ3_XXS, IQ3_S.
- Auto-select tiers based on VRAM & RAM.

### v1.2.0 (2026-10-14) — Unified model directory
- Strata-data moved into `models/LLM/Qwen3.8-Flash-Next/Strata-data/` (single download).
- Comprehensive bilingual README.

### v1.1.0
- Auto-start on demand (`ensure_running`), WinError 10035 blocking-socket fix.

### v1.0.0
- Initial release — health/start/setup web routes, 3 nodes, OpenAI-compatible proxy.
