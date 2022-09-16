from pulsar import Client, AuthenticationToken
from dotenv import load_dotenv
import os
import requests
import json
import itertools
from PIL import Image, ImageDraw, ImageFont
import logging as log
from maker import prompt_generator, prompt_list_creator

load_dotenv()
AI21_TOKEN = os.getenv("AI21_TOKEN")
PULSAR_URL = os.getenv("PULSAR_URL")
PULSAR_TOKEN = os.getenv("PULSAR_TOKEN")


def style_testing(prompt, to_test):
    prompts = []
    for word in range(len(to_test) + 1):
        for subset in itertools.combinations(to_test, word):
            prompts.append(prompt + " " + " ".join(subset))

    return prompts


def send_to_queue(prompts, steps, seed):
    for prompt in prompts:
        if prompt.endswith(" "):
            prompt = prompt[:-1]
        config = {"prompt": prompt, "seed": seed, "steps": steps}
        data = json.dumps(config)
        producer.send(data.encode("utf-8"))
        print("Sent to queue: " + data)
    producer.send("@@@stop@@@".encode("utf-8"))
    print("Sent to queue: @@@stop@@@")


def draw_description(text, width):
    new_text = ""
    for count, word in enumerate(text.split()):
        new_text += word
        if (count + 1) % 5 == 0:
            new_text += "\n"
        else:
            new_text += " "

    text = new_text[:-1]

    im = Image.new("RGB", (width, 150), "#fff")
    box = (10, 10, width - 10, 150 - 10)
    draw = ImageDraw.Draw(im)
    draw.rectangle(box, outline="#000")

    font_size = 100
    size = None
    while (
        size is None or size[0] > box[2] - box[0] or size[1] > box[3] - box[1]
    ) and font_size > 0:
        font = ImageFont.truetype("fonts/Roboto-Light.ttf", font_size)
        size = font.getsize_multiline(text)
        font_size -= 1
    draw.multiline_text((box[0], box[1]), text, "#000", font)

    return im


def generate_result(list_of_results):
    paths = [dic["files_path"][0] for dic in list_of_results]
    benchmark_prompts = [dic["prompt"] for dic in list_of_results]
    benchmark_width = list_of_results[0]["width"]

    images = [Image.open(x) for x in paths]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    text_images = [
        draw_description(prompt, benchmark_width) for prompt in benchmark_prompts
    ]
    text_widths, text_heights = zip(*(i.size for i in text_images))

    text_total_width = sum(text_widths)
    text_max_height = 150

    text_new_im = Image.new("RGB", (text_total_width, text_max_height))

    x_offset = 0
    for im in text_images:
        text_new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    def get_concat_v(im1, im2):
        dst = Image.new("RGB", (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    path = "benchmark.png"
    get_concat_v(text_new_im, new_im).save(path)

    config = {
        "prompt": list_of_results[0]["prompt"],
        "seed": list_of_results[0]["seed"],
        "steps": list_of_results[0]["prompt"],
        "files_path": [path],
    }

    print("Sending to discord the result of the benchmark")
    data = json.dumps(config)
    discord_producer.send(data.encode("utf-8"))


if __name__ == "__main__":
    pulsar_logger = log.getLogger("pulsar")
    pulsar_logger.setLevel(log.WARNING)

    client = Client(
        PULSAR_URL,
        authentication=AuthenticationToken(PULSAR_TOKEN),
        logger=pulsar_logger,
    )
    producer = client.create_producer("benchmark-prompt-queue")
    discord_producer = client.create_producer("result-post-discord")
    consumer = client.subscribe("benchmark-result", "benchmark-result-reading")

    prompt = "gay porn"
    to_test = ["digital art", "by James Gilleard", "trending on ArtStation"]
    steps = 25

    # Each will have a unique seed
    nb_of_tests = 10

    prompts = style_testing(prompt, to_test)

    for seed in range(nb_of_tests):
        send_to_queue(prompts, steps, seed)
        print("Waiting for benchmark results")
        msg = consumer.receive()
        print("Received benchmark results")
        data = msg.data().decode("utf-8")
        list_of_results = json.loads(data)
        print("Generating end results")
        generate_result(list_of_results)
        consumer.acknowledge(msg)

    client.close()
