import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


print("===================")
print("MODEL:",
      os.getenv("QWEN_MODEL"))

print("BASE URL:",
      os.getenv("DASHSCOPE_BASE_URL"))

key = os.getenv("DASHSCOPE_API_KEY")

print(
    "API KEY:",
    key[:8] if key else "None"
)


client = OpenAI(
    api_key=key,
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL"
    ),
    timeout=60
)


print("Calling Qwen...")


response = client.chat.completions.create(
    model=os.getenv("QWEN_MODEL"),
    messages=[
        {
            "role":"user",
            "content":"hello"
        }
    ],
    max_tokens=20
)


print("Response:")

print(
    response.choices[0].message.content
)