from pulsar import Client, AuthenticationToken
from dotenv import load_dotenv
import os
import requests
import json

load_dotenv()
AI21_TOKEN = os.getenv("AI21_TOKEN")
PULSAR_URL = os.getenv("PULSAR_URL")
PULSAR_TOKEN = os.getenv("PULSAR_TOKEN")

client = Client(PULSAR_URL, authentication=AuthenticationToken(PULSAR_TOKEN))
producer = client.create_producer("prompt-queue")


def prompt_testing():
    text_prompt = "stars and milky way"
    artist = "Damien Hirst"
    prompt = f'--prompt "{text_prompt}" --H 512 --W 512 --n_iter 1 --n_samples 2 --turbo --ddim_steps 25 --artist "{artist}"'
    producer.send(prompt.encode("utf-8"))
    client.close()


def prompt_generator(base_prompt, nb_prompts):
    result = requests.post(
        "https://api.ai21.com/studio/v1/j1-jumbo/complete",
        headers={"Authorization": f"Bearer {AI21_TOKEN}"},
        json={
            "prompt": base_prompt,
            "numResults": nb_prompts,
            "maxTokens": 26,
            "temperature": 0.7,
            "topKReturn": 0,
            "topP": 1,
            "countPenalty": {
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False,
            },
            "frequencyPenalty": {
                "scale": 0,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False,
            },
            "presencePenalty": {
                "scale": 1.58,
                "applyToNumbers": False,
                "applyToPunctuations": False,
                "applyToStopwords": False,
                "applyToWhitespaces": False,
                "applyToEmojis": False,
            },
            "stopSequences": ["↵", "."],
        },
    )

    return result


def prompt_list_creator(
    base_prompt, number_of_prompts_to_curate, prompt_list, prompt_extra, print_list=True
):

    curated_prompt_list = []
    current_number_of_prompts = number_of_prompts_to_curate
    base_data = json.loads(prompt_list.text)
    banned_character = [
        '"',
        "'",
        "(",
        ")",
        "_",
        "/",
    ]

    banned_words = [
        "perhaps",
        "or",
        "you",
        "I",
        "yourself",
        "he",
        "she",
        "etc",
    ]

    while len(curated_prompt_list) < number_of_prompts_to_curate:

        failed_prompts = 0

        print(f"Curating {current_number_of_prompts} prompts")

        for n in range(current_number_of_prompts):

            generated = base_data["completions"][n]["data"]["text"]

            if generated.startswith(" "):
                generated = generated[1:]

            failed_prompts += 1
            if generated == "":
                continue
            elif any(x in generated for x in banned_character):
                print(f"[BANNED]: {generated}")
                continue
            elif any(x in banned_words for x in generated.split(" ")):
                print(f"[BANNED]: {generated}")
                continue
            """elif any(char.isdigit() for char in generated):
                print(f"[BANNED]: {generated}")
                continue"""

            failed_prompts -= 1

            text_prompt = generated

            if prompt_extra["digital_art"]:
                text_prompt += ", digital art"

            curated_prompt_list.append(text_prompt)

        if failed_prompts != 0:
            print(f"{failed_prompts} prompts failed")
            result = prompt_generator(base_prompt, failed_prompts)
            base_data = json.loads(result.text)
            current_number_of_prompts = failed_prompts

    if print_list:
        print("\n\n\n")
        for prompt in curated_prompt_list:
            print(f"-> [{prompt}]")
        print("\n\n\n")

    return curated_prompt_list


def prompt_sender(prompt_list, parameters):

    for prompt in prompt_list:
        command = f'--prompt "{prompt}" --H {parameters["height"]} --W {parameters["width"]} --n_iter {parameters["iterations"]} --n_samples {parameters["samples"]} --ddim_steps {parameters["steps"]}'

        if parameters["turbo"]:
            command += " --turbo"

        if parameters["same_seed"]:
            command += " --same_seed"

        if parameters["random_artist"]:
            command += " --random_artist"

        if parameters["vary_artist"]:
            command += " --vary_artist"

        if "artist" in parameters:
            command += f' --artist "{parameters["artist"]}"'

        command += " --unet_bs " + str(parameters["samples"] * 2)

        try:
            producer.send(command.encode("utf-8"))
        except Exception as e:
            print(e)
            break


if __name__ == "__main__":

    base_prompt = "describe a brief mystical landscape scene:"
    n_of_prompts = 16  # 16 Max
    send_to_queue = True

    result = prompt_generator(base_prompt, n_of_prompts)

    prompt_extra = {"digital_art": True}

    prompt_list = prompt_list_creator(base_prompt, n_of_prompts, result, prompt_extra)

    parameters = {
        "height": 640,
        "width": 512,
        "iterations": 1,
        "samples": 2,
        "turbo": True,
        "steps": 50,
        "same_seed": False,
        "random_artist": False,
        "vary_artist": True,
        "artist": "James Gilleard",
    }

    if send_to_queue:
        prompt_sender(prompt_list, parameters)

    client.close()
