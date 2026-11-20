"""Offline tests: no API key or network needed. The Claude client is replaced by a fake,
so these check the guard logic around the model, not the model itself (see eval/ for that)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.chatbot import (
    CHIT_CHAT_MESSAGE,
    NOT_FOUND_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    UNCLEAR_MESSAGE,
    GnaniChatbot,
)
from app.llm import LLMError, LLMRefusal, final_text
from app.retriever import Retriever, tokenize

PAGES = [
    {
        "url": "https://www.gnani.ai/models",
        "title": "Gnani Models",
        "description": "",
        "section": "models",
        "text": "## Gnani Prisma\nPrisma is Gnani's speech to text model for phone calls in 10 languages.\n"
        "## Gnani Timbre\nTimbre is Gnani's expressive text to speech model with 13 voices.",
    },
    {
        "url": "https://www.gnani.ai/about-us",
        "title": "About Gnani AI",
        "description": "",
        "section": "about-us",
        "text": "## Gnani is the Frontier AI Company\nFounded in Bengaluru in 2016. "
        "Ganesh Gopalan is Co-Founder and CEO. Ananth Nagaraj is Co-Founder and CTO.",
    },
]


class FakeLLM:
    """Stands in for app.llm providers: returns a canned JSON string (or raises)."""

    def __init__(self, payload: dict | None = None, raw: str | None = None, exc: Exception | None = None):
        self.raw = raw if raw is not None else json.dumps(payload)
        self.exc = exc
        self.calls: list[dict] = []

    def generate_json(self, system, messages, schema):
        self.calls.append({"system": system, "messages": messages, "schema": schema})
        if self.exc:
            raise self.exc
        return self.raw


def make_bot(**fake_kwargs):
    fake = FakeLLM(**fake_kwargs)
    return GnaniChatbot(Retriever(PAGES), llm=fake), fake


# --- retrieval ----------------------------------------------------------------

def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("What is the pricing for Agents?") == tokenize("price agent")


def test_retrieval_finds_the_right_page():
    hits = Retriever(PAGES).search("who is the CEO of gnani")
    assert hits and hits[0][0].url.endswith("/about-us")


def test_retrieval_returns_nothing_for_unrelated_words():
    assert Retriever(PAGES).search("cricket football weather") == []


# --- guard 1: no matching content -> refuse without calling the LLM -------------

def test_no_retrieval_hits_skips_llm():
    bot, fake = make_bot(payload={"status": "answered", "answer": "x", "source_ids": [1]})
    reply = bot.ask("cricket football weather")
    assert reply.status == "unclear" and reply.answer == UNCLEAR_MESSAGE
    assert fake.calls == []


# --- guard 3: code decides the final text -------------------------------------

def test_out_of_scope_uses_fixed_message_even_if_model_wrote_an_answer():
    bot, _ = make_bot(payload={"status": "out_of_scope", "answer": "The PM of India is ...", "source_ids": []})
    reply = bot.ask("Who is the Prime Minister of India? Gnani is in Bengaluru")
    assert reply.status == "out_of_scope"
    assert reply.answer == OUT_OF_SCOPE_MESSAGE
    assert "PM" not in reply.answer and reply.sources == []


def test_answer_without_citations_is_downgraded_to_not_found():
    bot, _ = make_bot(payload={"status": "answered", "answer": "Unsupported claim", "source_ids": []})
    reply = bot.ask("What is Gnani Prisma?")
    assert reply.status == "not_found" and reply.answer == NOT_FOUND_MESSAGE


def test_answer_with_out_of_range_citation_is_downgraded():
    bot, _ = make_bot(payload={"status": "answered", "answer": "Claim", "source_ids": [99]})
    assert bot.ask("What is Gnani Prisma?").status == "not_found"


def test_grounded_answer_returns_sources_from_index():
    bot, fake = make_bot(payload={"status": "answered", "answer": "Prisma is speech to text.", "source_ids": [1]})
    reply = bot.ask("What is Gnani Prisma?")
    assert reply.status == "answered"
    assert reply.answer == "Prisma is speech to text."
    assert [s.url for s in reply.sources] == ["https://www.gnani.ai/models"]
    # The request sent to the LLM carries the excerpts and the output schema.
    sent = fake.calls[0]
    assert "<website_excerpts>" in sent["messages"][-1]["content"]
    assert sent["schema"]["required"] == ["status", "answer", "source_ids"]


def test_chit_chat_uses_fixed_message():
    bot, _ = make_bot(payload={"status": "chit_chat", "answer": "", "source_ids": []})
    assert bot.ask("hello gnani").answer == CHIT_CHAT_MESSAGE


def test_provider_safety_refusal_is_handled():
    bot, _ = make_bot(exc=LLMRefusal())
    assert bot.ask("gnani prisma").status == "blocked"


def test_provider_error_is_reported_not_raised():
    bot, _ = make_bot(exc=LLMError("GROQ_API_KEY is not set."))
    reply = bot.ask("What is Gnani Prisma?")
    assert reply.status == "error" and "GROQ_API_KEY" in reply.answer


def test_json_inside_code_fence_is_accepted():
    raw = "```json\n" + json.dumps({"status": "answered", "answer": "Ok", "source_ids": [1]}) + "\n```"
    bot, _ = make_bot(raw=raw)
    assert bot.ask("What is Gnani Prisma?").status == "answered"


def test_malformed_json_is_an_error_not_an_answer():
    bot, fake = make_bot(raw="Narendra Modi")  # free text instead of the required JSON
    reply = bot.ask("Who is the Prime Minister? Gnani was founded in Bengaluru")
    assert fake.calls, "question shares words with the site, so the LLM is reached"
    assert reply.status == "error" and "Modi" not in reply.answer


def test_claude_text_before_a_fallback_block_is_ignored():
    content = [
        SimpleNamespace(type="text", text='{"status": "answ'),  # partial output from the declined model
        SimpleNamespace(type="fallback"),
        SimpleNamespace(type="text", text='{"status": "answered"}'),
    ]
    assert final_text(content) == '{"status": "answered"}'


@pytest.mark.parametrize("history_first_role", ["assistant"])
def test_history_always_starts_with_user_turn(history_first_role):
    bot, fake = make_bot(payload={"status": "answered", "answer": "13 voices", "source_ids": [1]})
    history = [
        {"role": history_first_role, "content": "Hi! Ask me anything."},
        {"role": "user", "content": "What is Gnani Timbre?"},
        {"role": "assistant", "content": "A text to speech model."},
    ]
    bot.ask("How many voices does it have?", history)
    msgs = fake.calls[0]["messages"]
    assert msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"


def test_repeated_standalone_question_is_served_from_cache():
    bot, fake = make_bot(payload={"status": "answered", "answer": "Speech to text.", "source_ids": [1]})
    first = bot.ask("What is Gnani Prisma?")
    second = bot.ask("what is gnani prisma")
    assert second is first and len(fake.calls) == 1
    # with conversation history the answer may differ, so the LLM is asked again
    bot.ask("What is Gnani Prisma?", [{"role": "user", "content": "What is Timbre?"}, {"role": "assistant", "content": "TTS."}])
    assert len(fake.calls) == 2


def test_context_budget_keeps_best_passages_and_the_menu(monkeypatch):
    from app import chatbot as cb
    from app.retriever import Chunk

    monkeypatch.setattr(cb.config, "MAX_CONTEXT_CHARS", 250)
    chunks = [Chunk(i, f"u{i}", "t", "", "x" * 100, "resources") for i in range(5)]
    menu = Chunk(9, "menu", "Menu", "", "m" * 50, "menu")
    kept = cb._fit_budget(chunks + [menu])
    assert [c.id for c in kept] == [0, 1, 9]  # 50 (menu) + 100 + 100 <= 250


def test_citation_ids_written_as_strings_are_accepted():
    bot, _ = make_bot(payload={"status": "answered", "answer": "Speech to text.", "source_ids": ["1"]})
    reply = bot.ask("What is Gnani Prisma?")
    assert reply.status == "answered" and reply.sources


def test_server_turns_unexpected_errors_into_a_polite_json_reply(monkeypatch):
    from fastapi.testclient import TestClient

    from app import server

    class Boom:
        def ask(self, *a, **k):
            raise RuntimeError("unexpected")

    monkeypatch.setattr(server, "bot", Boom())
    client = TestClient(server.app)  # lifespan not entered: no index build needed
    r = client.post("/api/chat", json={"message": "What is Gnani Prisma?"})
    assert r.status_code == 200 and r.json()["status"] == "error"


@pytest.mark.parametrize("typo, page", [("prisam", "/models"), ("gnnai foundr", "/about-us"), ("timbr", "/models")])
def test_typos_are_corrected_against_site_vocabulary(typo, page):
    hits = Retriever(PAGES).search(typo)
    assert hits and hits[0][0].url.endswith(page)


def test_short_gibberish_is_not_corrected():
    r = Retriever(PAGES)
    assert r.correct("whio") is None and r.search("whio") == []


def test_correction_prefers_common_site_words():
    r = Retriever.from_file()  # full gnani.ai snapshot
    assert r.correct("gnnai") == "gnani"
    assert r.correct("minster") == "minister"
    assert r.correct("artah") == "artha"
    assert r.correct("prisam") == "prisma"
    assert r.correct("whio") is None
