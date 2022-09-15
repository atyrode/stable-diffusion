from pulsar import Client, AuthenticationToken
from dotenv import load_dotenv
import os
import requests
import json
import itertools

load_dotenv()
AI21_TOKEN = os.getenv("AI21_TOKEN")
PULSAR_URL = os.getenv("PULSAR_URL")
PULSAR_TOKEN = os.getenv("PULSAR_TOKEN")


def style_testing(prompt, artist):
    artist = f"by {artist}"
    to_test = ["digital art", artist, "trending on ArtStation"]

    prompts = []
    for word in range(len(to_test) + 1):
        for subset in itertools.combinations(to_test, word):
            prompts.append(prompt + " " + " ".join(subset))

    return prompts


def send_to_queue(prompts, steps, seed):
    for prompt in prompts:
        if prompt.endswith(" "):
            prompt = prompt[:-1]
        to_send = f'--prompt "{prompt}" --H 512 --W 512 --n_iter 1 --seed {seed} --n_samples 1 --turbo --ddim_steps {steps}'
        producer.send(to_send.encode("utf-8"))
        print("Sent to queue: " + to_send)
    producer.send("@@@stop@@@".encode("utf-8"))
    print("Sent to queue: @@@stop@@@")


if __name__ == "__main__":
    client = Client(PULSAR_URL, authentication=AuthenticationToken(PULSAR_TOKEN))
    producer = client.create_producer("benchmark-prompt-queue")

    prompt = "two drunk guys at the bar having sex"
    artist = "James Gilleard"
    steps = 50

    # Each will have a unique seed
    nb_of_tests = 2

    prompts = style_testing(prompt, artist)

    for seed in range(nb_of_tests):
        send_to_queue(prompts, steps, seed)

    client.close()
