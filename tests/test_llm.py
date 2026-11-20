"""Offline tests for the OpenAI-compatible provider (Groq / Ollama) error handling."""

from __future__ import annotations

from types import SimpleNamespace

import httpx2
import openai
import pytest

from app.llm import LLMError, OpenAICompatibleLLM


def validation_error() -> openai.BadRequestError:
    body = {"message": "Failed to validate JSON.", "code": "json_validate_failed", "failed_generation": ""}
    response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return openai.BadRequestError("Failed to validate JSON.", response=response, body=body)


def ok(text: str):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text))])


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make_llm(outcomes, fallback=""):
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "openai/gpt-oss-120b", fallback)
    fake = FakeCompletions(outcomes)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake), base_url="https://example.invalid/v1")
    return llm, fake


def test_json_validation_failure_is_retried():
    llm, fake = make_llm([validation_error(), ok('{"status": "out_of_scope", "answer": "", "source_ids": []}')])
    assert '"out_of_scope"' in llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})
    assert len(fake.calls) == 2


def test_repeated_validation_failures_fall_back_to_plain_text_then_give_up():
    llm, fake = make_llm([validation_error()] * 3)
    with pytest.raises(LLMError):
        llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})
    modes = [c.get("response_format", {}).get("type", "text") for c in fake.calls]
    assert modes == ["json_object", "json_object", "text"]


def test_request_too_large_gives_a_clear_error():
    response = httpx2.Response(413, request=httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    err = openai.APIStatusError("Request too large", response=response, body={})
    llm, _ = make_llm([err])
    with pytest.raises(LLMError, match="too large"):
        llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})


def test_gpt_oss_gets_low_reasoning_effort_and_tight_output_cap():
    llm, fake = make_llm([ok("{}")])
    llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})
    assert fake.calls[0]["reasoning_effort"] == "low"
    assert fake.calls[0]["max_completion_tokens"] == 1000


def rate_limit_error(message: str) -> openai.RateLimitError:
    response = httpx2.Response(429, request=httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions"))
    return openai.RateLimitError(message, response=response, body={"message": message})


def test_rate_limit_with_short_wait_is_retried(monkeypatch):
    slept = []
    monkeypatch.setattr("app.llm.time.sleep", slept.append)
    llm, fake = make_llm([rate_limit_error("Rate limit reached. Please try again in 1.5s."), ok("{}")])
    assert llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"}) == "{}"
    assert slept == [2.0] and len(fake.calls) == 2


def test_rate_limit_with_long_wait_is_reported():
    llm, _ = make_llm([rate_limit_error("Rate limit reached on tokens per day. Please try again in 12m3s.")])
    with pytest.raises(LLMError, match="rate limit"):
        llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})


def test_daily_quota_switches_to_fallback_model_and_stays_there():
    quota = rate_limit_error("Limit reached on tokens per day (TPD). Please try again in 9m39.7s.")
    llm, fake = make_llm([quota, ok("{}"), ok("{}")], fallback="openai/gpt-oss-20b")
    llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})
    llm.generate_json("sys", [{"role": "user", "content": "q2"}], {"type": "object"})
    assert [c["model"] for c in fake.calls] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-20b"]
