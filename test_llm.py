import os
from openai import OpenAI


api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise RuntimeError(
        "没有检测到 DEEPSEEK_API_KEY，请先设置环境变量。"
    )


client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)


response = client.chat.completions.create(
    model="deepseek-flash",
    messages=[
        {
            "role": "system",
            "content": "You are a concise financial research assistant."
        },
        {
            "role": "user",
            "content": "Reply with exactly: API connection successful"
        }
    ],
    reasoning_effort="none",
    stream=False
)


print(response.choices[0].message.content)
