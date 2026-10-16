# BSAI Strata 模型 API 调用文档

## 📡 基础信息

| 项目 | 值 |
|------|-----|
| API Base URL | `http://127.0.0.1:8080/v1` |
| API Key | 不需要（本地运行，随便填或留空） |
| 协议 | OpenAI 兼容格式 |
| 模型类型 | 多模态（文本 + 视觉） |

---

## 🤖 当前可用模型

切换模型后，模型名称会变化。可以通过 `/v1/models` 接口查询当前模型。

常见模型名称：
- `qwen3.8-flash-next-q2_0` - Q2_0 最快
- `qwen3.8-flash-next-iq2_xs` - IQ2_XS 平衡
- `qwen3.8-flash-next-iq3_xxs` - IQ3_XXS 更好
- `qwen3.8-flash-next-iq3_s` - IQ3_S 最好质量

---

## 🔌 支持的接口

### 1. 获取模型列表
```
GET /v1/models
```

### 2. 聊天补全（推荐）
```
POST /v1/chat/completions
```

### 3. 文本补全
```
POST /v1/completions
```

---

## 💻 调用示例

### Python 示例（OpenAI SDK）

```python
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed"  # 本地运行，随便填
)

# 聊天补全
response = client.chat.completions.create(
    model="qwen3.8-flash-next-q2_0",  # 用 /v1/models 查询当前模型名
    messages=[
        {"role": "user", "content": "你好，介绍一下你自己"}
    ],
    temperature=0.7,
    max_tokens=2048
)

print(response.choices[0].message.content)
```

### 多模态（图片理解）示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="qwen3.8-flash-next-q2_0",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片的内容"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{base64编码的图片}"}}
            ]
        }
    ],
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### cURL 示例

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next-q2_0",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 512
  }'
```

---

## ⚙️ 模型管理 API（切换模型用）

如果需要通过 API 切换模型，调用 8081 端口的管理接口：

### 获取当前模型
```
GET http://127.0.0.1:8081/api/current
```

响应示例：
```json
{"current": "IQ3_XXS"}
```

### 切换模型
```
POST http://127.0.0.1:8081/api/switch
Content-Type: application/json

{"model": "IQ3_XXS"}
```

> ⚠️ 切换模型会重启引擎，需要等待 1-2 分钟加载完成后才能继续调用。

---

## 📝 注意事项

1. **本地运行**：模型运行在本地，不需要联网
2. **首次加载慢**：第一次启动或切换模型需要 1-2 分钟加载模型到显存
3. **显存占用**：根据量化版本不同，占用 10-40GB 显存
4. **多模态**：支持图片输入，可以做图片反推、图像理解
5. **兼容性**：完全兼容 OpenAI API 格式，可以直接用 OpenAI SDK 调用

---

## 🎯 快速测试

启动 Strata 后，直接访问：
- Web UI: `http://127.0.0.1:8080/`
- API 文档: `http://127.0.0.1:8080/docs`（如果有）
- 模型列表: `http://127.0.0.1:8080/v1/models`
