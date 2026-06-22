import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag import answer, answer_stream, classify_intent, chitchat_reply

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)

sessions = {}


class Query(BaseModel):
    session_id: str
    question: str


def _sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/ask")
def ask(q: Query):
    history = sessions.setdefault(q.session_id, [])

    if classify_intent(q.question) == "chat":
        reply = chitchat_reply(q.question, history)
        history.extend([("human", q.question), ("ai", reply)])
        sessions[q.session_id] = history[-12:]
        return {"answer": reply, "sources": []}

    text, sources = answer(q.question, history)
    history.extend([("human", q.question), ("ai", text)])
    sessions[q.session_id] = history[-12:]
    return {"answer": text, "sources": sources}


@app.post("/ask/stream")
def ask_stream(q: Query):
    history = sessions.setdefault(q.session_id, [])

    def gen():
        if classify_intent(q.question) == "chat":
            reply = chitchat_reply(q.question, history)
            yield _sse({"type": "sources", "sources": []})
            yield _sse({"type": "token", "text": reply})
            history.extend([("human", q.question), ("ai", reply)])
            sessions[q.session_id] = history[-12:]
            yield _sse({"type": "done"})
            return

        parts = []
        for kind, payload in answer_stream(q.question, history):
            if kind == "sources":
                yield _sse({"type": "sources", "sources": payload})
            else:  # token
                parts.append(payload)
                yield _sse({"type": "token", "text": payload})

        full = "".join(parts)
        history.extend([("human", q.question), ("ai", full)])
        sessions[q.session_id] = history[-12:]
        yield _sse({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")