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

from app import config

log = logging.getLogger(__name__)


class LLMError(Exception):
    """A provider/configuration problem; the message is safe to show to the user."""


class LLMRefusal(Exception):
    """The provider's own safety system declined the request."""


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str):
        import openai  # imported here so the anthropic-only setup doesn't need it

        self._openai = openai
        self.model = model
        # Free tiers have per-minute token limits; the SDK waits (honouring retry-after)
        # and retries, so a busy minute means a slower answer rather than an error.
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=90.0, max_retries=6)
        self.use_json_schema = True  # downgraded to plain JSON mode if the model can't do schemas

    def generate_json(self, system: str, messages: list[dict], schema: dict) -> str:
        openai = self._openai
        chat = [{"role": "system", "content": system}, *messages]
        max_tokens = config.MAX_TOKENS
        extra = {"reasoning_effort": config.REASONING_EFFORT} if "gpt-oss" in self.model else {}
        validation_failures = 0
        use_schema = self.use_json_schema
        while True:
            if use_schema:
                fmt = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema, "strict": True}}
            else:
                fmt = {"type": "json_object"}
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=chat,
                    response_format=fmt,
                    temperature=0.1,
                    max_completion_tokens=max_tokens,
                    **extra,
                )
                if resp.choices[0].finish_reason == "length" and max_tokens == config.MAX_TOKENS:
                    max_tokens *= 3  # a rare long answer: retry once with more room
                    log.info("Output hit the %d-token cap; retrying with %d", config.MAX_TOKENS, max_tokens)
                    continue
                break
            except openai.BadRequestError as e:
                body = e.body if isinstance(e.body, dict) else {}
                if body.get("code") == "json_validate_failed" or "json_validate_failed" in str(e):
                    # Groq checks the output against the schema and occasionally rejects a
                    # generation (often an empty one). The same request usually succeeds on
                    # retry; after two failures, fall back to plain JSON mode for this call.
                    validation_failures += 1
                    log.warning("Output failed JSON validation (attempt %d); retrying", validation_failures)
                    if validation_failures == 2:
                        use_schema = False
                    if validation_failures <= 2:
                        continue
                    raise LLMError("Sorry, I couldn't produce a proper answer. Please try again.") from e
                if use_schema and ("json_schema" in str(e) or "response_format" in str(e)):
                    log.warning("%s doesn't support JSON schema output; using JSON mode", self.model)
                    self.use_json_schema = use_schema = False
                    continue
                raise LLMError(f"The language model rejected the request: {e.message}") from e
            except openai.AuthenticationError as e:
                raise LLMError("The API key is missing or invalid. Check GROQ_API_KEY in .env.") from e
            except openai.RateLimitError as e:
                raise LLMError("The free API rate limit was reached. Please wait a minute and try again.") from e
            except openai.NotFoundError as e:
                raise LLMError(f"Model '{self.model}' was not found. Set GNANI_MODEL in .env.") from e
            except openai.APIConnectionError as e:
                raise LLMError(f"Could not reach the language model at {self.client.base_url}.") from e
            except openai.APIStatusError as e:
                raise LLMError(f"The language model service returned an error ({e.status_code}).") from e

        if resp.usage:
            log.info("%s tokens: prompt=%s completion=%s", self.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
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
        return OpenAICompatibleLLM(config.LLM_BASE_URL or "https://api.groq.com/openai/v1", config.GROQ_API_KEY, model)
    if provider == "ollama":
        return OpenAICompatibleLLM(config.LLM_BASE_URL or "http://localhost:11434/v1", "ollama", model)
    if provider == "openai-compatible":
        return OpenAICompatibleLLM(config.LLM_BASE_URL, config.LLM_API_KEY, model)
    raise LLMError(f"Unknown GNANI_LLM_PROVIDER '{provider}'. Use groq, ollama, anthropic or openai-compatible.")
