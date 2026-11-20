"""FastAPI web server: serves the chat UI and a JSON chat endpoint.

Run:  uvicorn app.server:app --port 8000      then open http://localhost:8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import config
from app.chatbot import GnaniChatbot
from app.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC_DIR = config.ROOT_DIR / "static"
bot: GnaniChatbot | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global bot
    retriever = Retriever.from_file()
    bot = GnaniChatbot(retriever)
    logging.info("Indexed %d pages into %d chunks", len(retriever.pages), len(retriever.chunks))
    yield


app = FastAPI(title="Gnani.ai Website Assistant", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")  # favicon and icons


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=40)


class SourceOut(BaseModel):
    title: str
    url: str


class ChatResponse(BaseModel):
    status: str
    answer: str
    sources: list[SourceOut]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    # Sync handler: FastAPI runs it in a worker thread, so the blocking API call is fine.
    try:
        reply = bot.ask(req.message, [t.model_dump() for t in req.history])
    except Exception:  # never send the UI a bare 500; log it and answer politely
        logging.exception("Unhandled error while answering %r", req.message[:200])
        return {"status": "error", "answer": "Sorry, something went wrong. Please try again.", "sources": []}
    return reply.to_dict()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "pages": len(bot.retriever.pages),
        "chunks": len(bot.retriever.chunks),
        "llm_provider": config.LLM_PROVIDER,
        "model": config.MODEL or config.DEFAULT_MODELS.get(config.LLM_PROVIDER, ""),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    # Browsers and crawlers request /favicon.ico directly; serve the PNG version.
    return FileResponse(STATIC_DIR / "favicon-32.png", media_type="image/png")
