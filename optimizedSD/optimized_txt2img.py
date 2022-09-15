import argparse, os, re
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging
import sys
import shlex
import signal
import json
from pulsar import Client, AuthenticationToken
from dotenv import load_dotenv
import os
import random
import time

load_dotenv()
PULSAR_URL = os.getenv("PULSAR_URL")
PULSAR_TOKEN = os.getenv("PULSAR_TOKEN")


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


client = Client(PULSAR_URL, authentication=AuthenticationToken(PULSAR_TOKEN))
consumer = client.subscribe("prompt-queue", "prompt-queue-reading")
benchmark_consumer = client.subscribe(
    "benchmark-prompt-queue", "benchmark-prompt-queue-reading"
)

producer = client.create_producer("result-post-discord")
benchmark_producer = client.create_producer("benchmark-result-post-discord")

random.seed(time.time())

with open("artist_list.txt") as f:
    artist_list = f.readlines()

parsed_artist_list = []

for artist in artist_list:
    artist_info = artist.split("\t")

    if "N/A" in artist_info[1]:
        artist_info[1] = ""

    if "N/A" in artist_info[0]:
        artist_info[1] = ""

    parsed_info = f"{artist_info[1][:-1]} {artist_info[0]}"
    parsed_artist_list.append(parsed_info)


# from samplers import CompVisDenoiser
logging.set_verbosity_error()


def signal_handler(sig, frame):
    client.close()
    print("You pressed Ctrl+C!")
    sys.exit(0)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


config = "optimizedSD/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render",
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="outputs/txt2img-samples",
)
parser.add_argument(
    "--skip_grid",
    action="store_true",
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)
parser.add_argument(
    "--skip_save",
    action="store_true",
    help="do not save individual samples. For speed measurements.",
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--fixed_code",
    action="store_true",
    help="if enabled, uses the same starting code across samples ",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)
parser.add_argument(
    "--n_samples",
    type=int,
    default=5,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="specify GPU (cuda/cuda:0/cuda:1/...)",
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--unet_bs",
    type=int,
    default=1,
    help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
)
parser.add_argument(
    "--turbo",
    action="store_true",
    help="Reduces inference time on the expense of 1GB VRAM",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast",
)
parser.add_argument(
    "--format",
    type=str,
    help="output image format",
    choices=["jpg", "png"],
    default="png",
)
parser.add_argument(
    "--sampler",
    type=str,
    help="sampler",
    choices=["ddim", "plms"],
    default="plms",
)
parser.add_argument("--same_seed", action="store_true")
parser.add_argument("--random_artist", action="store_true")
parser.add_argument("--vary_artist", action="store_true")
parser.add_argument(
    "--artist",
    type=str,
)

opt = parser.parse_args()

# Logging
logger(vars(opt), log_csv="logs/txt2img_logs.csv")

sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, value in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.unet_bs = opt.unet_bs
model.cdevice = opt.device
model.turbo = opt.turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = opt.device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd

if opt.device != "cpu" and opt.precision == "autocast":
    model.half()
    modelCS.half()

print("Please state your prompt:")

list_of_path = []
while True:

    signal.signal(signal.SIGINT, signal_handler)

    # When @@@stop@@@ empty list above

    benchmark = True
    try:
        msg = benchmark_consumer.receive(timeout_millis=1000)
        benchmark_consumer.acknowledge(msg)

        if msg.data().decode("utf-8") == "@@@stop@@@":
            paths = [dic["path"] for dic in list_of_path]
            benchmark_prompts = [dic["prompt"] for dic in list_of_path]
            benchmark_width = list_of_path[0]["width"]
            benchmark_pulsar_data = f"[**{list_of_path[0]['prompt']}**]"
            benchmark_pulsar_data += f" Seed: {list_of_path[0]['seed']}"
            benchmark_pulsar_data += f" Steps: {list_of_path[0]['steps']}"

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
                draw_description(prompt, benchmark_width)
                for prompt in benchmark_prompts
            ]
            text_widths, text_heights = zip(*(i.size for i in text_images))

            text_total_width = sum(text_widths)
            text_max_height = 150

            text_new_im = Image.new("RGB", (text_total_width, text_max_height))

            # draw_description(text, width, height)

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

            benchmark_producer.send(benchmark_pulsar_data.encode("utf-8"))

            list_of_path = []
            continue

    except Exception as e:
        benchmark = False

    if not benchmark:
        try:
            msg = consumer.receive(timeout_millis=1000)
            consumer.acknowledge(msg)

        except Exception as e:
            continue

    prompt = msg.data().decode("utf-8")

    print(f"Your prompt is: {prompt}")

    try:
        opt = parser.parse_args(shlex.split(prompt))

        tic = time.time()
        os.makedirs(opt.outdir, exist_ok=True)
        outpath = opt.outdir
        grid_count = len(os.listdir(outpath)) - 1

        if opt.seed == None:
            opt.seed = randint(0, 1000000)
        seed_everything(opt.seed)

        start_code = None
        if opt.fixed_code:
            start_code = torch.randn(
                [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f],
                device=opt.device,
            )

        batch_size = opt.n_samples
        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
        if not opt.from_file:
            assert opt.prompt is not None
            prompt = opt.prompt
            data = [batch_size * [prompt]]

            # Bad code below, will probably break at some point when messing with parameters
            # due to how I access the data list
            if opt.random_artist:
                new_data = []
                rand_number = random.randint(0, len(parsed_artist_list))

                for i, prompt in enumerate(data[0]):
                    artist = parsed_artist_list[rand_number]
                    new_data.append(prompt + f" by {artist}")

                    if opt.vary_artist:
                        rand_number = random.randint(0, len(parsed_artist_list))

                data[0] = new_data

            elif opt.artist:
                new_data = []

                for i, prompt in enumerate(data[0]):
                    new_data.append(prompt + f" by {opt.artist}")

                data[0] = new_data

        else:
            print(f"reading prompts from {opt.from_file}")
            with open(opt.from_file, "r") as f:
                data = f.read().splitlines()
                data = batch_size * list(data)
                data = list(chunk(sorted(data), batch_size))

        if opt.precision == "autocast" and opt.device != "cpu":
            precision_scope = autocast
        else:
            precision_scope = nullcontext

        with torch.no_grad():

            all_samples = list()
            for n in trange(opt.n_iter, desc="Sampling"):
                for prompts in tqdm(data, desc="data"):

                    sample_path = os.path.join(
                        outpath, "_".join(re.split(":| ", prompts[0]))
                    )[:150]
                    os.makedirs(sample_path, exist_ok=True)
                    base_count = len(os.listdir(sample_path))

                    with precision_scope("cuda"):
                        modelCS.to(opt.device)
                        uc = None
                        if opt.scale != 1.0:
                            uc = modelCS.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)

                        subprompts, weights = split_weighted_subprompts(prompts[0])
                        if len(subprompts) > 1:
                            c = torch.zeros_like(uc)
                            totalWeight = sum(weights)
                            # normalize each "sub prompt" and add it
                            for i in range(len(subprompts)):
                                weight = weights[i]
                                # if not skip_normalize:
                                weight = weight / totalWeight
                                c = torch.add(
                                    c,
                                    modelCS.get_learned_conditioning(subprompts[i]),
                                    alpha=weight,
                                )
                        else:
                            c = modelCS.get_learned_conditioning(prompts)

                        shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]

                        if opt.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            modelCS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)

                        samples_ddim = model.sample(
                            S=opt.ddim_steps,
                            conditioning=c,
                            seed=opt.seed,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=opt.scale,
                            unconditional_conditioning=uc,
                            eta=opt.ddim_eta,
                            x_T=start_code,
                            sampler=opt.sampler,
                        )

                        modelFS.to(opt.device)

                        print(samples_ddim.shape)
                        print("saving images")
                        for i in range(batch_size):

                            x_samples_ddim = modelFS.decode_first_stage(
                                samples_ddim[i].unsqueeze(0)
                            )
                            x_sample = torch.clamp(
                                (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                            )
                            x_sample = 255.0 * rearrange(
                                x_sample[0].cpu().numpy(), "c h w -> h w c"
                            )

                            path = os.path.join(
                                sample_path,
                                "seed_"
                                + str(opt.seed)
                                + "_"
                                + f"{base_count:05}.{opt.format}",
                            )
                            Image.fromarray(x_sample.astype(np.uint8)).save(path)

                            if not opt.same_seed:
                                opt.seed += 1
                            base_count += 1

                            pulsar_data = {
                                "prompt": prompts[i],
                                "path": path,
                                "seed": opt.seed,
                                "steps": opt.ddim_steps,
                                "width": opt.W,
                                "height": opt.H,
                            }
                            pulsar_data_string = json.dumps(pulsar_data)

                            if benchmark:
                                list_of_path.append(pulsar_data)
                            else:
                                try:
                                    producer.send(pulsar_data_string.encode("utf-8"))
                                except Exception as e:
                                    print(e)

                        if opt.device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            modelFS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)
                        del samples_ddim
                        print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

        toc = time.time()

        time_taken = (toc - tic) / 60.0

        print(
            (
                "Samples finished in {0:.2f} minutes and exported to " + sample_path
            ).format(time_taken)
        )

        print("Please state your next prompt:")

    except Exception as e:
        # Pulsar message failed to be processed
        consumer.negative_acknowledge(msg)
        print(e)
        client.close()
        break
