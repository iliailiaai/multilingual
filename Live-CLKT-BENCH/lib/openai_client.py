import os
import time
import json
import re
from datetime import datetime
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemma-4-31b-it:free"
LOCAL_BASE_URL = "http://localhost:8000/v1"
LOCAL_MODEL = "local-model"
RETRY_WAIT_SECONDS = 10
DEFAULT_JSON_ERROR_DIR = "test_data/json_errors"


class GenerateOutput():
    def __init__(self, text: List[str]):
        self.text = text


class JSONParseError(ValueError):
    pass


class OpenAIModel:
    def __init__(
        self,
        model: str = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        api_key: str = None,
        provider: str = "openrouter",
        json_error_dir: str = DEFAULT_JSON_ERROR_DIR,
        json_retry: int = 3,
    ):
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.json_error_dir = json_error_dir
        self.json_retry = json_retry

        if provider == "local":
            self.model = model or LOCAL_MODEL
            base_url = LOCAL_BASE_URL
            api_key = api_key or os.getenv("LOCAL_OPENAI_API_KEY", "EMPTY")
        elif provider == "openrouter":
            self.model = model or OPENROUTER_MODEL
            base_url = OPENROUTER_BASE_URL
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
        else:
            raise ValueError("provider must be 'openrouter' or 'local'")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def generate(
        self,
        prompt: str,
        num_return_sequences: int = 1,
        retry: int = 30,
        response_format: dict = None,
        json_context: str = "generate",
    ) -> GenerateOutput:

        json_attempt = 0
        for i in range(1, retry + 1):
            try:
                messages = [{"role": "user", "content": prompt}]

                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "n": num_return_sequences,
                }

                if response_format and self.provider != "local":
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)

                texto = [choice.message.content for choice in response.choices]
                if response_format and self.provider == "local":
                    json_attempt += 1
                    try:
                        texto = [
                            self._normalize_json_text(text, context=json_context)
                            for text in texto
                        ]
                    except JSONParseError as e:
                        print(f"[JSON ERROR] {e}")
                        if json_attempt >= self.json_retry:
                            raise

                        print(
                            f"[JSON ERROR] Retrying local JSON generation "
                            f"({json_attempt}/{self.json_retry})..."
                        )
                        continue
                return GenerateOutput(text=texto)

            except JSONParseError:
                raise
            except Exception as e:
                print(
                    f"Error: {e}, retrying in {RETRY_WAIT_SECONDS}s "
                    f"({i}/{retry})..."
                )
                time.sleep(RETRY_WAIT_SECONDS)

        raise RuntimeError(f"Failed after {retry} retries")

    def _normalize_json_text(self, text: str, context: str):
        return json.dumps(self.parse_json(text, context=context), ensure_ascii=False)

    def parse_json(self, text: str, context: str):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        candidate = self._extract_json_candidate(text)
        if candidate is not None:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        path = self._record_json_error(text, context)
        preview = text.replace("\n", "\\n")[:500]
        print(f"[JSON ERROR] Context: {context}")
        print(f"[JSON ERROR] Raw preview: {preview}")
        print(f"[JSON ERROR] Full raw response saved to: {path}")
        raise JSONParseError(f"Failed to parse JSON for {context}; raw response saved to {path}")

    def _extract_json_candidate(self, text: str):
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()

        decoder = json.JSONDecoder()
        starts = [idx for idx, char in enumerate(text) if char in "{["]
        for idx in starts:
            try:
                _, end = decoder.raw_decode(text[idx:])
                return text[idx:idx + end]
            except json.JSONDecodeError:
                continue
        return None

    def _record_json_error(self, text: str, context: str):
        os.makedirs(self.json_error_dir, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
        safe_context = re.sub(r"[^A-Za-z0-9_.-]+", "_", context).strip("_")[:120]
        filename = f"{stamp}_{safe_context or 'json_error'}.txt"
        path = os.path.join(self.json_error_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path


if __name__ == "__main__":
    model = OpenAIModel(
        model=LOCAL_MODEL,
        temperature=0.8,
        max_tokens=9999
    )

    output = model.generate(
        prompt="Say this is a test. Output in json format with two fields: field1 and field2. Each field should contain a short sentence.",
        num_return_sequences=2,
        retry=3,
        response_format={"type": "json_object"}
    )

    print(output.text)
