from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import error, request

from serving.prompt_templates import RetrievedPassage, build_cited_rag_prompt


DEFAULT_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        names: list[str] = []
        for model in models:
            if isinstance(model, dict):
                name = model.get("name") or model.get("model")
                if name:
                    names.append(str(name))
        return names

    def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            "stream": False,
        }
        if options:
            payload["options"] = options
        if keep_alive:
            payload["keep_alive"] = keep_alive
        return self._request_json("POST", "/api/chat", payload)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        http_request = request.Request(
            url=f"{self.base_url}{path}",
            method=method,
            data=body,
            headers=headers,
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request failed with status {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama at "
                f"{self.base_url}. Start the Ollama app or daemon and verify the local API is reachable."
            ) from exc

        try:
            payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Ollama returned invalid JSON: {response_body[:200]}"
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Ollama returned an unexpected non-object JSON payload.")
        return payload


class OllamaGenerator:
    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        model: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        system_prompt: str | None = None,
        options: dict[str, Any] | None = None,
        keep_alive: str | None = None,
        strict_model_check: bool = True,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.options = options or {}
        self.keep_alive = keep_alive
        self.strict_model_check = strict_model_check
        self.client = OllamaClient(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )

    def validate_model(self) -> list[str]:
        models = self.client.list_models()
        if not self.strict_model_check:
            return models

        if self.model in models:
            return models

        available = ", ".join(models) if models else "(none)"
        raise RuntimeError(
            f"Configured Ollama model '{self.model}' is not currently available. "
            f"Pulled models: {available}"
        )

    def generate_grounded_answer(
        self,
        *,
        query: str,
        passages: list[RetrievedPassage],
        unknown_response: str = "I don't know.",
    ) -> dict[str, Any]:
        prompt = build_cited_rag_prompt(
            query,
            passages,
            unknown_response=unknown_response,
        )
        messages: list[ChatMessage] = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=self.options,
            keep_alive=self.keep_alive,
        )
        message = response.get("message", {})
        if not isinstance(message, dict):
            raise RuntimeError("Ollama chat response did not include a message object.")

        answer = message.get("content")
        if not isinstance(answer, str):
            raise RuntimeError("Ollama chat response did not include message.content text.")

        return {
            "prompt": prompt,
            "answer": answer.strip(),
            "raw_response": response,
        }
