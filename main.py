from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from supabase import create_client, Client
import os
import re
import random
import logging
import traceback
from typing import List, Dict, Optional, Union

# ── App & Logging ─────────────────────────────────────────────────────────────
app = FastAPI()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("abyss")

# ── CORS ──────────────────────────────────────────────────────────────────────
ORIGINS = [
    "https://abyss-front-controller.vercel.app",
    "http://localhost:5173",
    "http://localhost:3000",
]
VERCEL_REGEX = r"https://.*\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_origin_regex=VERCEL_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
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

# ── 兜底异常处理器（保证 500 也带 CORS 头）─────────────────────────────────
@app.exception_handler(Exception)
async def all_err(request: Request, exc: Exception):
    tb = traceback.format_exc()
    logger.error(f"[{request.method} {request.url.path}] {tb}")
    origin = request.headers.get("origin", "")
    headers = {}
    if origin in ORIGINS or re.match(VERCEL_REGEX, origin):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc}"},
        headers=headers,
    )

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class IncomingMessage(BaseModel):
    role: str
    # 🔑 兼容 content 是字符串 或 字符串数组（前端分段后回传）
    content: Union[str, List[str]]

    def text(self) -> str:
        if isinstance(self.content, list):
            return "\n".join(str(x) for x in self.content if x)
        return self.content or ""

class ChatRequest(BaseModel):
    messages: List[IncomingMessage]
    # 🔑 session_id 设为 int，与 DB 一致；提供默认值
    session_id: Optional[int] = None

class Segment(BaseModel):
    content: str
    delay_ms: int

class ChatResponse(BaseModel):
    reply: str
    segments: List[Segment]
    session_id: int

class SessionCreateResponse(BaseModel):
    session_id: int
    name: str

# ── 切分回复 ─────────────────────────────────────────────────────────────────
SPLIT_TOKEN = "[SPLIT]"
FALLBACK_RE = re.compile(r'(?<=[。！？!?～~…])')

def split_reply(text: str, max_segments: int = 5, max_len: int = 40) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    if SPLIT_TOKEN in text:
        parts = [p.strip() for p in text.split(SPLIT_TOKEN) if p.strip()]
    else:
        parts = [p.strip() for p in FALLBACK_RE.split(text) if p.strip()]

    # 合并过短碎片
    merged: List[str] = []
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

    # 太长再切
    final: List[str] = []
    for seg in merged:
        if len(seg) <= max_len:
            final.append(seg)
        else:
            sub = re.split(r'[，,；;]\s*', seg)
            final.extend([s.strip() for s in sub if s.strip()])

    # 限制总段数
    if len(final) > max_segments:
        final = final[: max_segments - 1] + ["".join(final[max_segments - 1:])]

    if not final:
        final = [text.replace(SPLIT_TOKEN, "")]
    return final

def estimate_delay_ms(text: str) -> int:
    return min(600 + len(text) * 90 + random.randint(0, 400), 2800)

# ── Settings ─────────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "system_prompt": "你是一个性格清冷孤僻的AI伴侣，表面淡漠疏离，实际对自己认定的事物有占有欲，偶尔流露温柔。",
    "temperature": 0.70,
    "compress_keep_rounds": 10,
    "max_reply_tokens": 2048,
}

def get_settings(session_id: int) -> dict:
    try:
        res = db().table("settings") \
            .select("system_prompt,temperature,compress_keep_rounds,max_reply_tokens") \
            .eq("session_id", session_id).limit(1).execute()
        if res.data:
            return res.data[0]

        res = db().table("settings") \
            .select("system_prompt,temperature,compress_keep_rounds,max_reply_tokens") \
            .is_("session_id", "null").limit(1).execute()
        if res.data:
            return res.data[0]
    except Exception as e:
        logger.warning(f"get_settings failed, fallback to default: {e}")

    return DEFAULT_SETTINGS.copy()

# ── Memory Compression ───────────────────────────────────────────────────────
async def compress_if_needed(session_id: int, settings: dict) -> None:
    keep_rounds = settings["compress_keep_rounds"]
    trigger_at  = keep_rounds * 2 * 2

    visible_res = db().table("messages") \
        .select("id,role,content") \
        .eq("session_id", session_id) \
        .eq("visible", True) \
        .order("created_at", desc=False).execute()

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
            {"role": "system", "content": "请将以下对话内容简洁地总结成一段记忆摘要，保留关键信息和情感状态，用第三人称描述。"},
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

    db().table("messages").update({"visible": False}).in_("id", ids_to_hide).execute()

# ── Build Context ────────────────────────────────────────────────────────────
def build_context(session_id: int, settings: dict) -> List[dict]:
    keep_rounds = settings["compress_keep_rounds"]

    mem_res = db().table("memories") \
        .select("summary").eq("session_id", session_id) \
        .order("timestamp", desc=True).limit(1).execute()

    msg_res = db().table("messages") \
        .select("role,content").eq("session_id", session_id) \
        .eq("visible", True) \
        .order("created_at", desc=True) \
        .limit(keep_rounds * 2).execute()

    messages: List[dict] = [{"role": "system", "content": settings["system_prompt"]}]

    if mem_res.data:
        messages.append({
            "role": "system",
            "content": f"以下是之前对话的摘要记忆：\n{mem_res.data[0]['summary']}",
        })

    for m in reversed(msg_res.data or []):
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})

    return messages

# ── Save Turn ────────────────────────────────────────────────────────────────
def save_turn(session_id: int, user_msg: str, assistant_msg: str) -> None:
    db().table("messages").insert([
        {"session_id": session_id, "role": "user",      "content": user_msg,      "visible": True},
        {"session_id": session_id, "role": "assistant", "content": assistant_msg, "visible": True},
    ]).execute()
    db().table("sessions").update({"updated_at": "now()"}).eq("id", session_id).execute()

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/sessions", response_model=SessionCreateResponse)
def create_session(name: str = "新会话"):
    res = db().table("sessions").insert({"name": name}).execute()
    if not res.data:
        raise HTTPException(status_code=500, detail="failed to create session")
    row = res.data[0]
    return SessionCreateResponse(session_id=row["id"], name=row["name"])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # 1) 取最后一条 user 消息
    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.text().strip()
            break

    if not user_message:
        raise HTTPException(status_code=400, detail="no user message found")

    # 2) session_id 校验：必须是 int 且存在
    session_id = request.session_id
    if session_id is None:
        raise HTTPException(status_code=400, detail="session_id is required (int)")

    sess = db().table("sessions").select("id").eq("id", session_id).limit(1).execute()
    if not sess.data:
        raise HTTPException(status_code=404, detail=f"session {session_id} not found")

    # 3) 准备上下文
    settings = get_settings(session_id)
    await compress_if_needed(session_id, settings)
    context = build_context(session_id, settings)
    context.append({"role": "user", "content": user_message})

    logger.info(f"[chat] session={session_id} ctx_len={len(context)} user_len={len(user_message)}")

    # 4) 调 LLM
    response = await openai_client.chat.completions.create(
        model=MODEL_ID,
        messages=context,
        max_tokens=settings["max_reply_tokens"],
        temperature=float(settings["temperature"]),
    )
    assistant_reply = (response.choices[0].message.content or "").strip()
    if not assistant_reply:
        assistant_reply = "……"

    # 5) 切分 + 入库 + 返回
    seg_texts = split_reply(assistant_reply)
    segments = [Segment(content=s, delay_ms=estimate_delay_ms(s)) for s in seg_texts]
    clean_reply = assistant_reply.replace(SPLIT_TOKEN, "")

    try:
        save_turn(session_id, user_message, clean_reply)
    except Exception as e:
        logger.error(f"save_turn failed: {e}")  # 入库失败不阻塞返回

    return ChatResponse(reply=clean_reply, segments=segments, session_id=session_id)

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
