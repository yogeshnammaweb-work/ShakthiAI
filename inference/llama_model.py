import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b-instruct"

class LlamaModel:
    def __init__(self):
        self.url = OLLAMA_URL
        self.model = MODEL_NAME

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=600,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed:\n{exc}"
            ) from exc

        data = response.json()

        output = data.get("response", "")

        if not isinstance(output, str):
            raise RuntimeError(
                f"Unexpected Ollama response:\n{data}"
            )

        output = output.strip()

        # Qwen3 may place reasoning and final answer
        # together inside the response field.
        if "</think>" in output.lower():
            output = re.split(
                r"</think>",
                output,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[1].strip()

        # Remove any remaining thinking tags.
        output = re.sub(
            r"</?think>",
            "",
            output,
            flags=re.IGNORECASE,
        ).strip()

        if not output:
            raise RuntimeError(
                f"Ollama returned an empty response:\n{data}"
            )

        return output