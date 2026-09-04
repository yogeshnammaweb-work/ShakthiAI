import re
import requests


class LlamaModel:
    def __init__(
        self,
        model: str = "gemma3:4b",
        url: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.url = url

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=300,
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

        # Remove reasoning if any model returns <think>...</think>
        output = re.sub(
            r"<think>.*?</think>",
            "",
            output,
            flags=re.DOTALL | re.IGNORECASE,
        )

        output = output.strip()

        if not output:
            raise RuntimeError(
                f"Ollama returned an empty response:\n{data}"
            )

        return output