# api_voice_only.py
import os
import json
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse

from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv("secrets.env")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Langfuse setup
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST
)
langfuse_handler = CallbackHandler()

# Model
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

# ----- Persistent Memory -----
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

class RedisStore:
    def __init__(self, url="redis://localhost:6379/0"):
        self.client = redis.Redis.from_url(url, decode_responses=True)

    def put(self, namespace, doc_id, value):
        ns_key = ":".join(namespace)
        self.client.hset(ns_key, doc_id, json.dumps(value))

    def get(self, namespace, doc_id):
        ns_key = ":".join(namespace)
        v = self.client.hget(ns_key, doc_id)
        return None if v is None else json.loads(v)

    def search(self, namespace):
        ns_key = ":".join(namespace)
        all_items = self.client.hgetall(ns_key)
        return [{"key": k, "value": json.loads(v)} for k, v in all_items.items()]

import shelve
class ShelveStore:
    def __init__(self, filename="memory_voice.shelf"):
        self.filename = filename

    def put(self, namespace, doc_id, value):
        ns_key = ":".join(namespace)
        with shelve.open(self.filename) as db:
            ns = db.get(ns_key, {})
            ns[doc_id] = value
            db[ns_key] = ns

    def get(self, namespace, doc_id):
        ns_key = ":".join(namespace)
        with shelve.open(self.filename) as db:
            ns = db.get(ns_key, {})
            return ns.get(doc_id)

    def search(self, namespace):
        ns_key = ":".join(namespace)
        with shelve.open(self.filename) as db:
            ns = db.get(ns_key, {})
            return [{"key": k, "value": v} for k, v in ns.items()]

if REDIS_AVAILABLE and os.getenv("USE_REDIS", "0") == "1":
    store = RedisStore(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
else:
    store = ShelveStore()

# ----- FastAPI App -----
app = FastAPI(title="mAIstro Voice API")

# Keep per-connection chat history
conversation_buffers = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")
            thread_id = data.get("thread_id", "default")
            user_id = data.get("user_id", "anonymous")

            if msg_type == "transcript":
                transcript = data.get("transcript", "")
                buffer = conversation_buffers.setdefault(thread_id, [])
                buffer.append(HumanMessage(content=transcript))

                try:
                    response = model.invoke(buffer)
                    ai_content = response.content
                except Exception as e:
                    ai_content = f"Error: {e}"

                buffer.append(AIMessage(content=ai_content))
                await websocket.send_text(json.dumps({"type": "assistant", "content": ai_content}))

            elif msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))

            else:
                await websocket.send_text(json.dumps({"type": "error", "message": "Unknown type"}))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("WebSocket error:", e)
        await websocket.close()

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    try:
        profile = [m["value"] for m in store.search(("profile", user_id))]
        todos = [m["value"] for m in store.search(("todo", user_id))]
        instructions = [m["value"] for m in store.search(("instructions", user_id))]
        return {"profile": profile, "todos": todos, "instructions": instructions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Serve voice client directly
@app.get("/voice")
async def get_voice_client():
    return FileResponse(os.path.join("static", "voice_client.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
