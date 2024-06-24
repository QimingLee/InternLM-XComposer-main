# python3
# Please install OpenAI SDK first：`pip3 install openai`
from openai import OpenAI

client = OpenAI(api_key="sk-4vfdkP4cczqNHUZtBbE655309785453f9d55A428EdFc3aBe", base_url="https://openkey.cloud/v1")

response = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)