# AI Companion Chat API

基于 FastAPI + OpenRouter + Supabase 的清冷 AI 伴侣后端。

## 文件结构

```
.
├── main.py                 # FastAPI 主应用
├── requirements.txt        # Python 依赖
├── render.yaml             # Render 部署配置
├── supabase_migration.sql  # Supabase 建表 SQL
└── .env.example            # 环境变量示例
```

## 快速开始

### 1. 创建 Supabase 表

在 Supabase 控制台 → SQL Editor 中执行 `supabase_migration.sql`。

### 2. 本地开发

```bash
cp .env.example .env
# 填写 .env 中的各项值

pip install -r requirements.txt
uvicorn main:app --reload
```

### 3. 部署到 Render

1. 将项目推送到 GitHub 仓库
2. 在 Render 控制台选择 **New → Web Service**，关联该仓库
3. Render 会自动读取 `render.yaml` 配置
4. 在 Render 的 **Environment** 页面填写以下环境变量：
   - `OPENROUTER_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `MODEL_ID`（可选，默认 `openai/gpt-4o-mini`）

## API

### POST /chat

**请求体**
```json
{
  "user_id": "user_123",
  "message": "你好"
}
```

**响应**
```json
{
  "reply": "……"
}
```

### GET /health

健康检查，返回 `{"status": "ok"}`。

## 记忆压缩逻辑

- 每次请求使用：**最新 summary + 最近 10 轮对话** 作为上下文
- 当非摘要记录超过 40 条（20 轮）时，自动将最早 20 条（10 轮）总结为一条 summary 存入 `memories` 表，并删除原始记录
