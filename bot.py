import discord
from dotenv import load_dotenv
import os
import asyncio


load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

class Bot(discord.Client):

    async def on_ready(self):
        print(f'Logged on as {self.user}!')
        
        channel = self.get_channel(1019007726455619615)
        await channel.send("test", file=discord.File('outputs/txt2img-samples/kitty_cat/seed_27_00000.png'))

    async def on_message(self, message):
        print(f'Message from {message.author}: {message.content}')
    
if __name__ == '__main__':
    intents = discord.Intents.default()
    intents.message_content = True

    bot = Bot(intents=intents)
    bot.run(DISCORD_TOKEN)

