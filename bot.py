import discord
import os
from dotenv import load_dotenv
import asyncio
# from openai import OpenAI


load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
# gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

import transformers
import torch
from transformers import AutoTokenizer

model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

logs = []

async def ask_question(chat_history, question):
    # Construct the message sequence
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": question})
    
    # response = gpt_client.chat.completions.create(
    #     messages=messages,
    #     model="gpt-3.5-turbo",
    # )

    # return response.choices[0].message.content

    sequences = pipeline(
        messages,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        truncation = True,
        max_length=400,
    )
    return sequences[0]["generated_text"][-1]["content"]

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # with open("log.txt", "a") as f:
    #     f.write(message.content + "\n")

    if message.content.startswith('question: '):
        query = " ".join(message.content.split(" ")[1:])
        answer = await ask_question(logs, query)
        await message.channel.send(answer)
    else:
        logs.append({'role': 'user', 'content': f"{message.author.display_name} {message.content}"})

client.run(os.environ.get("token"))
