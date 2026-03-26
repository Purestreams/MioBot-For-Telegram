import json
import sys
import requests
from app.runtime_config import get_runtime_value


class ChatClient:
    def __init__(self, api_key: str | None = None, url: str | None = None):
        self.api_key = api_key or get_runtime_value("AZURE_API_KEY") or get_runtime_value("AZURE_OPENAI_API_KEY")
        self.url = url or get_runtime_value("AZURE_API_URL") or get_runtime_value("AZURE_OPENAI_ENDPOINT")
        if not self.api_key:
            raise ValueError("API key is required. Set AZURE_OPENAI_API_KEY or AZURE_API_KEY.")
        if not self.url:
            raise ValueError(
                "Endpoint URL is required. Set AZURE_OPENAI_ENDPOINT or AZURE_API_URL, "
                "or pass url explicitly."
            )
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def chat(
        self,
        messages: list[dict],
        model: str = "gpt-oss-120b",
        max_completion_tokens: int = 2048,
        temperature: float = 0.5,
        top_p: float | None = None,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        timeout: int = 60,
    ) -> str:
        payload = {
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "model": model,
        }
        if top_p is not None:
            payload["top_p"] = top_p

        resp = requests.post(self.url, headers=self.headers, json=payload, timeout=timeout)
        resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError:
            return resp.text

        choices = (data or {}).get("choices") or []
        if choices:
            first = choices[0] or {}
            content = ((first.get("message") or {}).get("content") or first.get("content"))
            if content:
                return content.strip()
        return json.dumps(data, indent=2)


def main():
    try:
        client = ChatClient(
            api_key="",
            url="",
        )
        content = client.chat(
            messages=[{"role": "user", "content": "what is machine learning?"}],
            temperature=0.5,
        )
        print(content)
    except requests.HTTPError as e:
        print(f"HTTP error: {e} - {getattr(e.response, 'text', '')}", file=sys.stderr)
        sys.exit(2)
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(3)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()