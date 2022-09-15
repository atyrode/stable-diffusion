from pulsar import Client, AuthenticationToken
from dotenv import load_dotenv
import os


load_dotenv()
AI21_TOKEN = os.getenv("AI21_TOKEN")
PULSAR_URL = os.getenv("PULSAR_URL")
PULSAR_TOKEN = os.getenv("PULSAR_TOKEN")

client = Client(PULSAR_URL, authentication=AuthenticationToken(PULSAR_TOKEN))

consumer = client.subscribe("benchmark-prompt-queue", "benchmark-prompt-queue-reading")

while True:
    msg = consumer.receive(timeout_millis=1000)
    try:
        print("Emptyied queue of: '{}' id='{}'".format(msg.data(), msg.message_id()))
        # Acknowledge successful processing of the message
        consumer.acknowledge(msg)
    except Exception:
        # Message failed to be processed
        consumer.negative_acknowledge(msg)

client.close()
