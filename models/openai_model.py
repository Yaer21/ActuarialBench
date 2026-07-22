"""
OpenAI Models (GPT-4o, GPT-4, GPT-3.5, etc.)
"""
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .base_model import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI API Model"""

    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1", **kwargs):
        self.reasoning_effort = kwargs.pop("reasoning_effort", None)
        self.thinking_level = kwargs.pop("thinking_level", None)
        self.thinking_tokens = kwargs.pop("thinking_tokens", None)
        self.enable_thinking = kwargs.pop("enable_thinking", None)
        self.thinking_type = kwargs.pop("thinking_type", None)
        self.output_effort = kwargs.pop("output_effort", None)
        self.chat_template_kwargs = kwargs.pop("chat_template_kwargs", None)
        self.api_mode = kwargs.pop("api_mode", None)
        self.force_stream = kwargs.pop("force_stream", False)
        super().__init__(model_name, api_key, base_url, **kwargs)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=float(self.timeout or 600.0),
        )

    def supports_spreadsheet_attachments(self) -> bool:
        base = (self.base_url or "").lower().rstrip("/")
        return base in {"https://api.openai.com/v1", "https://api.openai.com"}

    def spreadsheet_attachment_support_message(self) -> str:
        if self.supports_spreadsheet_attachments():
            return ""
        return (
            "Direct `.xlsx` attachment is only implemented for the official OpenAI "
            f"Files/Responses API. Current endpoint: {self.base_url}"
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Call OpenAI API to generate response

        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt, uses default ACTUARIAL_SYSTEM_PROMPT if not provided
            **kwargs: Additional arguments

        Returns:
            Generated text from the model
        """
        if kwargs.get("spreadsheet_attachment_path"):
            details = self.generate_with_details(prompt, system_prompt=system_prompt, **kwargs)
            return details.get("final_response", "")

        sys_prompt = system_prompt if system_prompt is not None else self.ACTUARIAL_SYSTEM_PROMPT

        if self.api_mode == "responses":
            return self._generate_with_responses_api(prompt, system_prompt=sys_prompt, **kwargs)["final_response"]
        if self.force_stream:
            return self._generate_with_streaming_chat(prompt, system_prompt=sys_prompt, **kwargs)["final_response"]

        try:
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": False,
            }

            reasoning_effort = kwargs.get("reasoning_effort") or getattr(self, "reasoning_effort", None)
            if reasoning_effort:
                completion_params["reasoning_effort"] = reasoning_effort

            extra_body = self._build_extra_body(kwargs)
            if extra_body:
                completion_params["extra_body"] = extra_body

            response = self.client.chat.completions.create(**completion_params)
            self.last_input_tokens = self._extract_input_tokens_from_response(response)
            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

    def generate_with_details(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """Best-effort native separation for OpenAI-compatible responses."""

        if kwargs.get("spreadsheet_attachment_path"):
            return self._generate_with_spreadsheet_attachment(prompt, system_prompt=system_prompt, **kwargs)

        sys_prompt = system_prompt if system_prompt is not None else self.ACTUARIAL_SYSTEM_PROMPT

        if self.api_mode == "responses":
            return self._generate_with_responses_api(prompt, system_prompt=sys_prompt, **kwargs)
        if self.force_stream:
            return self._generate_with_streaming_chat(prompt, system_prompt=sys_prompt, **kwargs)

        try:
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": False,
            }

            reasoning_effort = kwargs.get("reasoning_effort") or getattr(self, "reasoning_effort", None)
            if reasoning_effort:
                completion_params["reasoning_effort"] = reasoning_effort

            extra_body = self._build_extra_body(kwargs)
            if extra_body:
                completion_params["extra_body"] = extra_body

            response = self.client.chat.completions.create(**completion_params)
            input_tokens = self._extract_input_tokens_from_response(response)
            self.last_input_tokens = input_tokens
            msg = response.choices[0].message

            content = getattr(msg, "content", None)
            raw = (content or "").strip()

            def _get_extra_field(key: str):
                if hasattr(msg, key):
                    return getattr(msg, key)
                if hasattr(msg, "model_dump"):
                    try:
                        data = msg.model_dump()
                        return data.get(key)
                    except Exception:
                        return None
                if isinstance(msg, dict):
                    return msg.get(key)
                return None

            reasoning = ""
            for key in ("reasoning", "thinking", "reasoning_content", "analysis"):
                value = _get_extra_field(key)
                if value:
                    reasoning = str(value).strip()
                    break

            if reasoning:
                return {"raw_response": raw, "reasoning": reasoning, "final_response": raw, "input_tokens": input_tokens}

            reasoning, final_text = self.split_reasoning_and_final(raw)
            return {"raw_response": raw, "reasoning": reasoning, "final_response": final_text, "input_tokens": input_tokens}
        except Exception as e:
            raise Exception(f"OpenAI API call failed: {str(e)}")

    def _build_extra_body(self, kwargs) -> dict:
        """Build provider/proxy-specific thinking controls for OpenAI-compatible APIs."""

        extra_body = {}
        thinking_type = kwargs.get("thinking_type") or getattr(self, "thinking_type", None)
        thinking_tokens = kwargs.get("thinking_tokens") or getattr(self, "thinking_tokens", None)
        output_effort = kwargs.get("output_effort") or getattr(self, "output_effort", None)
        thinking_level = kwargs.get("thinking_level") or getattr(self, "thinking_level", None)
        enable_thinking = kwargs.get("enable_thinking") if "enable_thinking" in kwargs else getattr(self, "enable_thinking", None)
        chat_template_kwargs = kwargs.get("chat_template_kwargs") or getattr(self, "chat_template_kwargs", None)

        if chat_template_kwargs:
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        if thinking_type:
            extra_body["thinking"] = {"type": thinking_type}
            if thinking_tokens and thinking_type != "adaptive":
                extra_body["thinking"]["budget_tokens"] = thinking_tokens
        elif enable_thinking is not None:
            extra_body["thinking"] = {
                "type": "enabled" if enable_thinking else "disabled"
            }
            if thinking_tokens and enable_thinking:
                extra_body["thinking"]["budget_tokens"] = thinking_tokens

        if output_effort or thinking_level:
            extra_body["output_config"] = {
                "effort": output_effort or thinking_level
            }

        return extra_body

    def _generate_with_spreadsheet_attachment(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        if not self.supports_spreadsheet_attachments():
            raise Exception(self.spreadsheet_attachment_support_message())

        attachment_path = Path(str(kwargs["spreadsheet_attachment_path"])).resolve()
        sys_prompt = system_prompt if system_prompt is not None else self.ACTUARIAL_SYSTEM_PROMPT
        uploaded_file = None

        try:
            with attachment_path.open("rb") as file_obj:
                uploaded_file = self.client.files.create(
                    file=file_obj,
                    purpose="user_data",
                )

            response = self.client.responses.create(
                model=self.model_name,
                instructions=sys_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_file", "file_id": uploaded_file.id},
                        ],
                    }
                ],
                tools=[
                    {
                        "type": "code_interpreter",
                        "container": {
                            "type": "auto",
                            "file_ids": [uploaded_file.id],
                        },
                    }
                ],
            )

            raw = self._response_output_text(response).strip()
            input_tokens = self._extract_input_tokens_from_response(response)
            self.last_input_tokens = input_tokens
            reasoning, final_text = self.split_reasoning_and_final(raw)
            return {
                "raw_response": raw,
                "reasoning": reasoning,
                "final_response": final_text,
                "input_tokens": input_tokens,
                "attachment_file_id": uploaded_file.id,
                "attachment_name": attachment_path.name,
            }
        except Exception as e:
            raise Exception(f"OpenAI Files/Responses API call failed: {str(e)}")
        finally:
            if uploaded_file is not None:
                try:
                    self.client.files.delete(uploaded_file.id)
                except Exception:
                    pass

    def _generate_with_responses_api(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        sys_prompt = system_prompt if system_prompt is not None else self.ACTUARIAL_SYSTEM_PROMPT

        try:
            response_params = {
                "model": self.model_name,
                "instructions": sys_prompt,
                "input": prompt,
                "temperature": kwargs.get("temperature", self.temperature),
            }

            reasoning_effort = kwargs.get("reasoning_effort") or getattr(self, "reasoning_effort", None)
            if reasoning_effort:
                response_params["reasoning"] = {"effort": reasoning_effort}

            response = self.client.responses.create(**response_params)
            raw = self._response_output_text(response).strip()
            input_tokens = self._extract_input_tokens_from_response(response)
            self.last_input_tokens = input_tokens
            reasoning, final_text = self.split_reasoning_and_final(raw)
            return {
                "raw_response": raw,
                "reasoning": reasoning,
                "final_response": final_text,
                "input_tokens": input_tokens,
            }
        except Exception as e:
            raise Exception(f"OpenAI Responses API call failed: {str(e)}")

    def _generate_with_streaming_chat(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        sys_prompt = system_prompt if system_prompt is not None else self.ACTUARIAL_SYSTEM_PROMPT

        try:
            completion_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                "temperature": kwargs.get("temperature", self.temperature),
                "stream": True,
            }

            extra_body = self._build_extra_body(kwargs)
            if extra_body:
                completion_params["extra_body"] = extra_body

            chunks = self.client.chat.completions.create(**completion_params)
            parts = []
            input_tokens = None
            for chunk in chunks:
                chunk_tokens = self._extract_input_tokens_from_response(chunk)
                if chunk_tokens is not None:
                    input_tokens = chunk_tokens

                choices = getattr(chunk, "choices", None) or []
                for choice in choices:
                    delta = getattr(choice, "delta", None)
                    content = getattr(delta, "content", None) if delta is not None else None
                    if content:
                        parts.append(content)

            raw = "".join(parts).strip()
            self.last_input_tokens = input_tokens
            reasoning, final_text = self.split_reasoning_and_final(raw)
            return {
                "raw_response": raw,
                "reasoning": reasoning,
                "final_response": final_text,
                "input_tokens": input_tokens,
            }
        except Exception as e:
            raise Exception(f"OpenAI streaming chat call failed: {str(e)}")

    def _response_output_text(self, response) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        parts = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text_value = getattr(content, "text", None)
                if isinstance(text_value, str) and text_value:
                    parts.append(text_value)
        return "\n".join(parts).strip()
