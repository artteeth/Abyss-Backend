from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
from supabase import create_client, Client
import os
import re
import random
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
from typing import List, Dict

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # 注意是 messages，复数
    session_id: str = "default"

# 【新增】一小段气泡的数据结构
class Segment(BaseModel):
    content: str        # 这段文字
    delay_ms: int       # 显示前要等多少毫秒（模拟打字时间）

class ChatResponse(BaseModel):
    reply: str                  # 完整回复（已去掉 [SPLIT]）
    segments: list[Segment]     # 【新增】切好的气泡列表
    session_id: int

class SessionCreateResponse(BaseModel):
    session_id: int
    name: str

# ── 【新增】切分回复的小工具 ──────────────────────────────────────────────
SPLIT_TOKEN = "[SPLIT]"
# 兜底切分：按句号、问号、叹号等切
FALLBACK_RE = re.compile(r'(?<=[。！？!?～~…])')

def split_reply(text: str, max_segments: int = 5, max_len: int = 40) -> list[str]:
    """
    把 AI 的整段回复切成几小段。
      1) 优先按 [SPLIT] 切
      2) 没有 [SPLIT] 就按句末标点切（兜底）
      3) 合并过短碎片（避免 "嗯。" 单独成段）
      4) 单段太长就按逗号再切
      5) 最多 max_segments 段
    """
    text = text.strip()
    if not text:
        return []

    # 1) 优先 [SPLIT]
    if SPLIT_TOKEN in text:
        parts = [p.strip() for p in text.split(SPLIT_TOKEN) if p.strip()]
    else:
        parts = [p.strip() for p in FALLBACK_RE.split(text) if p.strip()]

    # 2) 合并太短的
    merged: list[str] = []
    buf = ""
    for p in parts:
        if len(buf) + len(p) <= 5:
            buf += p
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)

    # 3) 太长的再切
    final: list[str] = []
    for seg in merged:
        if len(seg) <= max_len:
            final.append(seg)
        else:
            sub = re.split(r'[，,；;]\s*', seg)
            final.extend([s.strip() for s in sub if s.strip()])

    # 4) 限制总段数：超出的并到最后一段
    if len(final) > max_segments:
        final = final[: max_segments - 1] + ["".join(final[max_segments - 1:])]

    # 5) 兜底：如果什么都没切出来，就返回原文
    if not final:
        final = [text.replace(SPLIT_TOKEN, "")]

    return final

def estimate_delay_ms(text: str) -> int:
    """
    估算这一段"显示前要等多久"。
    600ms 起步 + 每字 90ms + 随机抖动 0~400ms，上限 2800ms。
    """
    base = 600
    per_char = 90
    jitter = random.randint(0, 400)
    return min(base + len(text) * per_char + jitter, 2800)

# ── Helpers: settings ─────────────────────────────────────────────────────────
def get_settings(session_id: int) -> dict:
    res = db().table("settings") \
        .select("system_prompt,temperature,compress_keep_rounds,max_reply_tokens") \
        .eq("session_id", session_id) \
        .limit(1).execute()
    if res.data:
        return res.data[0]

    res = db().table("settings") \
        .select("system_prompt,temperature,compress_keep_rounds,max_reply_tokens") \
        .is_("session_id", "null") \
        .limit(1).execute()
    if res.data:
        return res.data[0]

    return {
        "system_prompt": "你是一个性格清冷孤僻的AI伴侣，表面淡漠疏离，实际对自己认定的事物有占有欲，偶尔流露温柔。",
        "temperature": 0.70,
        "compress_keep_rounds": 10,
        "max_reply_tokens": 2048,
    }

# ── Helpers: memory compression ───────────────────────────────────────────────
async def compress_if_needed(session_id: int, settings: dict) -> None:
    keep_rounds = settings["compress_keep_rounds"]
    trigger_at  = keep_rounds * 2 * 2

    visible_res = db().table("messages") \
        .select("id,role,content") \
        .eq("session_id", session_id) \
        .eq("visible", True) \
        .order("created_at", desc=False) \
        .execute()

    records = visible_res.data or []
    if len(records) <= trigger_at:
        return

    to_compress = records[: keep_rounds * 2]
    ids_to_hide = [r["id"] for r in to_compress]

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

    db().table("memories").insert({
        "session_id": session_id,
        "summary": summary_text,
        "metadata": {"compressed_message_count": len(to_compress)},
    }).execute()

    db().table("messages") \
        .update({"visible": False}) \
        .in_("id", ids_to_hide) \
        .execute()

# ── Helpers: build context ────────────────────────────────────────────────────
def build_context(session_id: int, settings: dict) -> list[dict]:
    keep_rounds = settings["compress_keep_rounds"]

    mem_res = db().table("memories") \
        .select("summary") \
        .eq("session_id", session_id) \
        .order("timestamp", desc=True) \
        .limit(1).execute()

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
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    return messages

# ── Helpers: save messages ────────────────────────────────────────────────────
def save_turn(session_id: int, user_msg: str, assistant_msg: str) -> None:
    db().table("messages").insert([
        {"session_id": session_id, "role": "user",      "content": user_msg,      "visible": True},
        {"session_id": session_id, "role": "assistant", "content": assistant_msg, "visible": True},
    ]).execute()

    db().table("sessions") \
        .update({"updated_at": "now()"}) \
        .eq("id", session_id).execute()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/sessions", response_model=SessionCreateResponse)
def create_session(name: str = "新会话"):
    res = db().table("sessions").insert({"name": name}).execute()
    row = res.data[0]
    return SessionCreateResponse(session_id=row["id"], name=row["name"])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id   = request.session_id
    user_message = None
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            user_message = msg.get("content", "").strip()
            break
            
    if user_message is None:
        return {"error": "没有找到用户消息"}

    sess = db().table("sessions").select("id").eq("id", session_id).limit(1).execute()
    if not sess.data:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")

    settings = get_settings(session_id)
    await compress_if_needed(session_id, settings)

    context = build_context(session_id, settings)
    context.append({"role": "user", "content": user_message})

    response = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=context,
        max_tokens=settings["max_reply_tokens"],
        temperature=float(settings["temperature"]),
    )
    assistant_reply = response.choices[0].message.content.strip()

    # ── 【新增的关键三步】 ────────────────────────────────────────────
    # 1) 切成多段
    seg_texts = split_reply(assistant_reply)
    segments = [Segment(content=s, delay_ms=estimate_delay_ms(s)) for s in seg_texts]

    # 2) 入库前把 [SPLIT] 删掉，存干净文本
    clean_reply = assistant_reply.replace(SPLIT_TOKEN, "")
    save_turn(session_id, user_message, clean_reply)

    # 3) 返回给前端
    return ChatResponse(
        reply=clean_reply,
        segments=segments,
        session_id=session_id,
    )

@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: int, visible_only: bool = True):
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
