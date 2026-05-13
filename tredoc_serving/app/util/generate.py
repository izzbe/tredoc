import os
import re
from openai import AsyncOpenAI

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = "tredoc"

client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="none")

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def extract_signature(source: str) -> str | None:
    pattern = re.compile(
        r"^((?:@\w[\w.]*(?:\(.*?\))?\s*\n)*)"  # optional decorators
        r"([ \t]*def\s+\w+\s*\(.*?\))"  # def name(params)
        r"(\s*->\s*[^:]+?)?"  # optional return annotation
        r"\s*:",  # colon
        re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return None
    return ((m.group(1) or "") + m.group(2) + (m.group(3) or "")).strip()


async def generate(code: str, style: str) -> str:
    signature = extract_signature(code)
    user_content = (
        f"Create a docstring for python code following the specifications: "
        f"<style>{style}</style>\n"
        f"<signature>{signature}</signature>\n"
        f"<body>{code}</body>"
    )
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=2048,
        temperature=0,
    )
    return response.choices[0].message.content
