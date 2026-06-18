from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import create_client, Client
import os
from typing import Optional

app = FastAPI(title="AI Companion Chat API")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Env / Clients ─────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID           = os.getenv("MODEL_ID", "openai/gpt-4o-mini")
SUPABASE_URL       = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY       = os.getenv("SUPABASE_KEY", "")


openai_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

_supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY)
    if SUPABASE_URL and SUPABASE_KEY else None
)

def db() -> Client:
    if not _supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    return _supabase


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str="default"
    message: str

class ChatResponse(BaseModel):
    reply: str
    session_id: int

class SessionCreateResponse(BaseModel):
    session_id: int
    name: str


# ── Helpers: settings ─────────────────────────────────────────────────────────
def get_settings(session_id: int) -> dict:
    """
    优先取会话专属配置，不存在则回退到全局默认（session_id IS NULL）。
    返回字段: system_prompt, temperature, compress_keep_rounds, max_reply_tokens
    """
    # 会话专属
    res = db().table("settings") \
        .select("system_prompt,temperature,compress_keep_rounds,max_reply_tokens") \
        .eq("session_id", session_id) \
        .limit(1).execute()
    if res.data:
        return res.data[0]

    # 全局默认
    res = db().table("settings") \
        .select("system_prompt,temperature,compress_keep_rounds,max_reply_tokens") \
        .is_("session_id", "null") \
        .limit(1).execute()
    if res.data:
        return res.data[0]

    # 兜底硬编码
    return {
        "system_prompt": "你是一个性格清冷孤僻的AI伴侣，表面淡漠疏离，实际对自己认定的事物有占有欲，偶尔流露温柔。",
        "temperature": 0.70,
        "compress_keep_rounds": 10,
        "max_reply_tokens": 2048,
    }


# ── Helpers: memory compression ───────────────────────────────────────────────
async def compress_if_needed(session_id: int, settings: dict) -> None:
    """
    当 visible=TRUE 的消息超过 compress_keep_rounds*2 条时，
    将最早的 compress_keep_rounds 轮（compress_keep_rounds*2 条）
    压缩为一条 memories 摘要，并把那些消息标记为 visible=FALSE。
    """
    keep_rounds = settings["compress_keep_rounds"]   # 保留最近 N 轮
    trigger_at  = keep_rounds * 2 * 2               # 超过 2×keep 轮才触发

    visible_res = db().table("messages") \
        .select("id,role,content") \
        .eq("session_id", session_id) \
        .eq("visible", True) \
        .order("created_at", desc=False) \
        .execute()

    records = visible_res.data or []
    if len(records) <= trigger_at:
        return

    # 取最早 keep_rounds 轮（keep_rounds*2 条）
    to_compress = records[: keep_rounds * 2]
    ids_to_hide  = [r["id"] for r in to_compress]

    # 生成摘要
    conversation_text = "\n".join(
        f"{r['role'].upper()}: {r['content']}" for r in to_compress
    )
    summary_resp = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "system",
                "content": "请将以下对话内容简洁地总结成一段记忆摘要，保留关键信息和情感状态，用第三人称描述。",
            },
            {"role": "user", "content": conversation_text},
        ],
        max_tokens=500,
    )
    summary_text = summary_resp.choices[0].message.content.strip()

    # 写入 memories 表
    db().table("memories").insert({
        "session_id": session_id,
        "summary": summary_text,
        "metadata": {"compressed_message_count": len(to_compress)},
    }).execute()

    # 将已压缩消息标记为不可见（软删除，保留原始数据）
    db().table("messages") \
        .update({"visible": False}) \
        .in_("id", ids_to_hide) \
        .execute()


# ── Helpers: build context ────────────────────────────────────────────────────
def build_context(session_id: int, settings: dict) -> list[dict]:
    """
    构建发送给模型的 messages 列表：
      system_prompt + [最新摘要] + 最近 keep_rounds 轮可见消息
    """
    keep_rounds = settings["compress_keep_rounds"]

    # 最新摘要
    mem_res = db().table("memories") \
        .select("summary") \
        .eq("session_id", session_id) \
        .order("timestamp", desc=True) \
        .limit(1).execute()

    # 最近可见消息（keep_rounds 轮 = keep_rounds*2 条）
    msg_res = db().table("messages") \
        .select("role,content") \
        .eq("session_id", session_id) \
        .eq("visible", True) \
        .order("created_at", desc=True) \
        .limit(keep_rounds * 2).execute()

    messages: list[dict] = [
        {"role": "system", "content": settings["system_prompt"]}
    ]

    if mem_res.data:
        messages.append({
            "role": "system",
            "content": f"以下是之前对话的摘要记忆：\n{mem_res.data[0]['summary']}",
        })

    recent = list(reversed(msg_res.data or []))
    for m in recent:
        # messages 表 role 包含 'tool'，OpenAI API 不接受 tool role 无 tool_call_id，跳过
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    return messages


# ── Helpers: save messages ────────────────────────────────────────────────────
def save_turn(session_id: int, user_msg: str, assistant_msg: str) -> None:
    db().table("messages").insert([
        {"session_id": session_id, "role": "user",      "content": user_msg,       "visible": True},
        {"session_id": session_id, "role": "assistant", "content": assistant_msg,  "visible": True},
    ]).execute()

    # 更新 session updated_at
    db().table("sessions") \
        .update({"updated_at": "now()"}) \
        .eq("id", session_id).execute()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/sessions", response_model=SessionCreateResponse)
def create_session(name: str = "新会话"):
    """创建新会话，返回 session_id"""
    res = db().table("sessions").insert({"name": name}).execute()
    row = res.data[0]
    return SessionCreateResponse(session_id=row["id"], name=row["name"])


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id  = request.session_id
    user_message = request.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="message is required")

    # 确认 session 存在
    sess = db().table("sessions").select("id").eq("id", session_id).limit(1).execute()
    if not sess.data:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")

    # 读取配置
    settings = get_settings(session_id)

    # 压缩旧对话（如需要）
    await compress_if_needed(session_id, settings)

    # 构建上下文 + 当前消息
    context = build_context(session_id, settings)
    context.append({"role": "user", "content": user_message})

    # 调用模型
    response = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=context,
        max_tokens=settings["max_reply_tokens"],
        temperature=float(settings["temperature"]),
    )
    assistant_reply = response.choices[0].message.content.strip()

    # 持久化
    save_turn(session_id, user_message, assistant_reply)

    return ChatResponse(reply=assistant_reply, session_id=session_id)


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: int, visible_only: bool = True):
    """查询某会话的消息历史"""
    q = db().table("messages") \
        .select("id,role,content,created_at,visible,reasoning_content") \
        .eq("session_id", session_id) \
        .order("created_at", desc=False)
    if visible_only:
        q = q.eq("visible", True)
    res = q.execute()
    return {"session_id": session_id, "messages": res.data or []}


@app.get("/health")
def health():
    return {"status": "ok"}
