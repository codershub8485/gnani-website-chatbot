"""LLM providers behind one tiny interface: generate_json(system, messages, schema) -> str.

  groq      (default) free API for open-source models, needs a free GROQ_API_KEY
  ollama    any model running locally with Ollama, no key at all
  anthropic Claude, needs ANTHROPIC_API_KEY

groq and ollama both speak the OpenAI-compatible chat API, so they share one class
(any other OpenAI-compatible server works too, via GNANI_LLM_BASE_URL).
The scope guards in chatbot.py are identical whichever provider is used.
"""

from __future__ import annotations

import logging
import re
import time

from app import config

log = logging.getLogger(__name__)


class LLMError(Exception):
    """A provider/configuration problem; the message is safe to show to the user."""


class LLMRefusal(Exception):
    """The provider's own safety system declined the request."""


class OpenAICompatibleLLM:
    # Output modes, tried in order. Plain JSON mode comes first on purpose: Groq's strict
    # JSON-schema mode returns an empty generation (400 json_validate_failed) for gpt-oss
    # whenever the answer contains characters such as non-breaking hyphens ("speech‑to‑text"),
    # while JSON mode returns well-formed JSON for the same prompt. The schema is still
    # enforced: it is in the system prompt, and chatbot.py validates every reply in code.
    MODES = ("json_object", "text")

    def __init__(self, base_url: str, api_key: str, model: str, fallback_model: str = ""):
        import openai  # imported here so the anthropic-only setup doesn't need it

        self._openai = openai
        self.model = model
        self.fallback_model = fallback_model if fallback_model and fallback_model != model else ""
        self._primary_blocked_until = 0.0  # set when the main model's quota runs out
        # Free tiers have per-minute token limits; the SDK waits (honouring retry-after)
        # and retries, so a busy minute means a slower answer rather than an error.
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=90.0, max_retries=6)
        self.mode = 0  # index into MODES; moves forward if a provider rejects a mode

    def generate_json(self, system: str, messages: list[dict], schema: dict) -> str:
        openai = self._openai
        chat = [{"role": "system", "content": system}, *messages]
        max_tokens = config.MAX_TOKENS
        mode, validation_failures, rate_limit_waits = self.mode, 0, 0
        use_fallback = self.fallback_model and time.time() < self._primary_blocked_until
        while True:
            model = self.fallback_model if use_fallback else self.model
            extra = {"reasoning_effort": config.REASONING_EFFORT} if "gpt-oss" in model else {}
            fmt = {"response_format": {"type": "json_object"}} if self.MODES[mode] == "json_object" else {}
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=chat,
                    temperature=0.1,
                    max_completion_tokens=max_tokens,
                    **fmt,
                    **extra,
                )
                if resp.choices[0].finish_reason == "length" and max_tokens == config.MAX_TOKENS:
                    max_tokens = 2 * config.MAX_TOKENS  # a rare long answer: retry once with more room
                    log.info("Output hit the %d-token cap; retrying with %d", config.MAX_TOKENS, max_tokens)
                    continue
                break
            except openai.BadRequestError as e:
                body = e.body if isinstance(e.body, dict) else {}
                if body.get("code") == "json_validate_failed" or "json_validate_failed" in str(e):
                    # The provider rejected malformed JSON. Retry once, then without JSON mode
                    # (the reply is parsed and validated in chatbot.py either way).
                    validation_failures += 1
                    log.warning("Output failed JSON validation (attempt %d); retrying", validation_failures)
                    if validation_failures >= 2 and mode + 1 < len(self.MODES):
                        mode += 1
                    if validation_failures <= 2:
                        continue
                    raise LLMError("Sorry, I couldn't produce a proper answer. Please try again.") from e
                if "response_format" in str(e) and mode + 1 < len(self.MODES):
                    log.warning("%s doesn't support JSON mode; using plain text", self.model)
                    self.mode = mode = mode + 1
                    continue
                raise LLMError(f"The language model rejected the request: {e.message}") from e
            except openai.AuthenticationError as e:
                raise LLMError("The API key is missing or invalid. Check GROQ_API_KEY in .env.") from e
            except openai.RateLimitError as e:
                # The SDK already retried what it could. Groq sometimes asks callers not to
                # retry but says when to come back ("try again in 7.2s"); wait that long once
                # or twice so the user sees a slower answer instead of an error.
                wait = _retry_after_seconds(e)
                rate_limit_waits += 1
                if (wait is None or wait > 45) and self.fallback_model and not use_fallback:
                    # Main model's quota is used up (e.g. the daily limit): switch models.
                    self._primary_blocked_until = time.time() + (wait or 600)
                    log.warning("%s is rate limited for %ss; using %s", self.model, wait, self.fallback_model)
                    use_fallback = True
                    continue
                if wait is not None and wait <= 45 and rate_limit_waits <= 3:
                    log.info("Rate limited; waiting %.1fs before retrying", wait)
                    time.sleep(wait + 0.5)
                    continue
                raise LLMError("The free API rate limit was reached. Please wait a minute and try again.") from e
            except openai.NotFoundError as e:
                raise LLMError(f"Model '{self.model}' was not found. Set GNANI_MODEL in .env.") from e
            except openai.APIConnectionError as e:
                raise LLMError(f"Could not reach the language model at {self.client.base_url}.") from e
            except openai.APIStatusError as e:
                if e.status_code == 413:
                    raise LLMError("That request was too large for the free API tier. Please ask a shorter question.") from e
                raise LLMError(f"The language model service returned an error ({e.status_code}).") from e

        if resp.usage:
            log.info("%s tokens: prompt=%s completion=%s", model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise LLMRefusal()
        return choice.message.content or ""


class ClaudeLLM:
    def __init__(self, model: str):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:  # created lazily so the app can start without a key
            self._client = self._anthropic.Anthropic(timeout=90.0, max_retries=2)
        return self._client

    def generate_json(self, system: str, messages: list[dict], schema: dict) -> str:
        anthropic = self._anthropic
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=config.CLAUDE_MAX_TOKENS,
                system=system,
                messages=messages,
                output_config={"effort": config.EFFORT, "format": {"type": "json_schema", "schema": schema}},
                # If a safety classifier declines, retry server-side on Anthropic's
                # recommended fallback model instead of failing the turn.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as e:
            raise LLMError("The Anthropic API key is missing or invalid. Set ANTHROPIC_API_KEY in .env.") from e
        except anthropic.RateLimitError as e:
            raise LLMError("The assistant is busy (rate limited). Please try again in a moment.") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"The language model service returned an error ({e.status_code}).") from e
        except anthropic.APIConnectionError as e:
            raise LLMError("Could not reach the language model service. Check your network.") from e
        except anthropic.AnthropicError as e:  # e.g. no credentials configured at all
            raise LLMError(f"Language model not configured: {e}") from e

        if response.stop_reason == "refusal":
            raise LLMRefusal()
        return final_text(response.content)


def _retry_after_seconds(err) -> float | None:
    """Seconds to wait, from the retry-after header or a "try again in 1m2.5s" message."""
    try:
        header = err.response.headers.get("retry-after")
        if header:
            return float(header)
    except (AttributeError, ValueError):
        pass
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)", str(err))
    if not m:
        return None
    minutes, value, unit = int(m.group(1) or 0), float(m.group(2)), m.group(3)
    return minutes * 60 + (value / 1000 if unit == "ms" else value)


def final_text(content) -> str:
    """Text of a Claude response. If a server-side fallback happened mid-response, only the
    text after the last `fallback` block belongs to the model that finished the answer."""
    texts: list[str] = []
    for block in content:
        if block.type == "fallback":
            texts = []
        elif block.type == "text":
            texts.append(block.text)
    return "".join(texts)


def make_llm():
    provider = config.LLM_PROVIDER
    model = config.MODEL or config.DEFAULT_MODELS.get(provider)
    if provider == "anthropic":
        return ClaudeLLM(model)
    if provider == "groq":
        if not config.GROQ_API_KEY:
            raise LLMError("GROQ_API_KEY is not set. Get a free key at https://console.groq.com/keys and put it in .env.")
        return OpenAICompatibleLLM(
            config.LLM_BASE_URL or "https://api.groq.com/openai/v1", config.GROQ_API_KEY, model, config.FALLBACK_MODEL
        )
    if provider == "ollama":
        return OpenAICompatibleLLM(config.LLM_BASE_URL or "http://localhost:11434/v1", "ollama", model)
    if provider == "openai-compatible":
        return OpenAICompatibleLLM(config.LLM_BASE_URL, config.LLM_API_KEY, model)
    raise LLMError(f"Unknown GNANI_LLM_PROVIDER '{provider}'. Use groq, ollama, anthropic or openai-compatible.")
