from openai import OpenAI

client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = "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"
)

completion = client.chat.completions.create(
    model="google/gemma-2-2b-it",
    messages=[{"role":"user","content":"Give a recipe of a meal or food."}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=True
)

for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

