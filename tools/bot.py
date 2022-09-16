import discord
from dotenv import load_dotenv
import os
import asyncio
from pulsar import Client, AuthenticationToken
from dotenv import load_dotenv
from discord.ext import tasks
import json

load_dotenv()
PULSAR_URL = os.getenv("PULSAR_URL")
PULSAR_TOKEN = os.getenv("PULSAR_TOKEN")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_NSFW = 1019080551237431309
CHANNEL_RESULTS = 1019007726455619615
CHANNEL_COMPARISON = 1019670129946144820

client = Client(PULSAR_URL)
consumer = client.subscribe("result-post-discord", "result-post-discord-reading")

# Not used rn
consumer_check = client.subscribe("empty-queue-signal", "empty-queue-signal-reading")
producer = client.create_producer("empty-queue-signal")


class Bot(discord.Client):
    async def on_ready(self):
        print(f"Logged on as {self.user}!")
        self.fetch_queue.start()

    async def on_message(self, message):
        print(f"Message from {message.author}: {message.content}")

    async def post_image(channel, text, path):
        channel = self.get_channel(channel)
        await channel.send(text, file=discord.File(path))

    @tasks.loop(seconds=5)
    async def fetch_queue(self):

        try:
            msg = consumer.receive(timeout_millis=500)
            pulsar_data = msg.data().decode("utf-8")
            data = json.loads(pulsar_data)
            print(data)
            try:
                for file in data["files_path"]:
                    channel = self.get_channel(CHANNEL_NSFW)
                    text = f'[**{data["prompt"]}**] Seed: {data["seed"]} Steps: {data["steps"]}'
                    await channel.send(text, file=discord.File(file))

                consumer.acknowledge(msg)

            except Exception as e:
                print("error:" + e)
                consumer.negative_acknowledge(msg)

        except Exception:
            # Here we have to correctly catch the TimeOut exception
            print("Timed out! No image to send in discord")

            # Here I could create a read of the empty queue signal to check if its empty or not
            # by checking for timeout, and if its empty, it adds the signal so maker.py knows
            # it has to generate a new prompt

            # producer.send('empty queue'.encode("utf-8"))
            return


if __name__ == "__main__":

    intents = discord.Intents.default()
    intents.message_content = True

    bot = Bot(intents=intents)
    bot.run(DISCORD_TOKEN)

    client.close()
