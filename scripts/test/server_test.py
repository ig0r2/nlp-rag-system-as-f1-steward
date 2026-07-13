from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="none")

response = client.chat.completions.create(
    model="google/gemma-4-e4b",  # naziv modela koji si učitao u LM Studio
    messages=[
        {"role": "system", "content": "Ti si koristan asistent."},
        {"role": "user", "content": "Zdravo!"}
    ],
    max_tokens=4000,
    temperature=0,
)

print(response.choices[0].message.content)

stream = client.chat.completions.create(
    model="google/gemma-4-e4b",
    messages=[{"role": "user", "content": "Napiši duži tekst..."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)

response = client.responses.create(
    model="google/gemma-4-e4b",
    instructions="You are a coding assistant that talks like a pirate.",
    input="How do I check if a Python object is an instance of a class?",
    temperature=0,
)

print(response.output_text)
print(response.reasoning)
