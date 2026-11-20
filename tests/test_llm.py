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


def make_llm(outcomes):
    llm = OpenAICompatibleLLM("https://example.invalid/v1", "key", "openai/gpt-oss-120b")
    fake = FakeCompletions(outcomes)
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=fake), base_url="https://example.invalid/v1")
    return llm, fake


def test_json_validation_failure_is_retried():
    llm, fake = make_llm([validation_error(), ok('{"status": "out_of_scope", "answer": "", "source_ids": []}')])
    assert '"out_of_scope"' in llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})
    assert len(fake.calls) == 2


def test_repeated_validation_failures_fall_back_to_json_mode_then_give_up():
    llm, fake = make_llm([validation_error()] * 3)
    with pytest.raises(LLMError):
        llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})
    assert [c["response_format"]["type"] for c in fake.calls] == ["json_schema", "json_schema", "json_object"]


def test_gpt_oss_gets_low_reasoning_effort_and_tight_output_cap():
    llm, fake = make_llm([ok("{}")])
    llm.generate_json("sys", [{"role": "user", "content": "q"}], {"type": "object"})
    assert fake.calls[0]["reasoning_effort"] == "low"
    assert fake.calls[0]["max_completion_tokens"] == 1000
