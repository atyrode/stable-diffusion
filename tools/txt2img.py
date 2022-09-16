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
from optimizedSD.optimUtils import split_weighted_subprompts, logger
from transformers import logging as t_logging
import sys
import shlex
import signal
import json
from pulsar import Client, AuthenticationToken
from dotenv import load_dotenv
import os
import random
import time
import logging as log
import atexit

# Two modes:
# pulsar = True the object holds and wait for pulsar input
# pulsar = False the object reads all prompts


class Txt2Img:
    def __init__(self, parameters={}, use_pulsar=False):

        self.use_pulsar = use_pulsar
        print(f"[Txt2Img.__init__]> Pulsar set to be used: {use_pulsar}")

        self.parameters = parameters
        self.sanitize_parameters()
        print("[Txt2Img.__init__]> Parameters sanitized")

        random.seed(time.time())
        print("[Txt2Img.__init__]> Random seeded with current time")

        t_logging.set_verbosity_error()
        print("[Txt2Img.__init__]> Logger from CompVisDenoiser set to log errors")

        self.load_model()
        print("[Txt2Img.__init__]> Loaded model from config")

        atexit.register(self.cleanup)

        if self.use_pulsar:
            self.init_pulsar()
            self.pulsar_loop()

    def sanitize_parameters(self):

        load_dotenv()

        default_parameters = {
            "model": "model.ckpt",
            "config": "v1-inference.yaml",
            "pulsar_url": None,
            "pulsar_token": None,
            "unet_bs": 1,  # Should ideally be n_sample x2 for the prompts you'll be running, slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )
            "device": "cuda",  # Specify GPU (cuda/cuda:0/cuda:1/...)
            "turbo": True,  # Reduces inference time on the expense of 750MB-1GB VRAM
            "precision": "autocast",  # autocast or full
        }

        for param in default_parameters:
            if param not in self.parameters:
                self.parameters[param] = default_parameters[param]

    def sanitize_config(self, config):

        default_config = {
            "prompt": "a chihuahua jumping",  # The prompt to render
            "save_at": "txt2img_output",  # If specified, it will save the image rendered at the given dir
            "grid": False,  # If true, it will also save the batch as a grid
            "steps": 50,  # The number of sampling steps for the prompt to use
            "fixed_code": True,  # (Unsure) Uses the same starting code across samples
            "ddim_eta": 0.0,  # ddim eta (eta=0.0 corresponds to deterministic sampling)
            "iterations": 1,  # Number of batches
            "samples": 1,  # Amount of image per batches
            "height": 512,  # Height in pixel
            "width": 512,  # Width in pixel
            "latent_channels": 4,  # (unsure)
            "downsampling_factors": 8,  # (unsure)
            "grid_rows": 0,  # Rows in the grid (default: n_samples)
            "scale": 7.5,  # (unsure) Unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))
            "seed": None,  # The seed (for reproducible sampling)
            "format": "png",  # Image format (png or jpg)
            "sampler": "plms",  # (unsure) ["ddim", "plms","heun", "euler", "euler_a", "dpm2", "dpm2_a", "lms"]
        }

        # Check config param type agaisnt default config?
        for param in default_config:
            if param not in config:
                config[param] = default_config[param]

        return config

    def init_pulsar(self):

        self.pulsar_logger = log.getLogger("pulsar")
        self.pulsar_logger.setLevel(log.WARNING)

        self.pulsar_client = Client(
            self.parameters["pulsar_url"],
            authentication=AuthenticationToken(self.parameters["pulsar_url"]),
            logger=self.pulsar_logger,
        )

        print("[Txt2Img.init_pulsar]> Pulsar client loaded")
        print(
            "[Txt2Img.init_pulsar]> You may ignore a failed connection on port 6650 (unknown bug reason)"
        )

        self.consumer = self.pulsar_client.subscribe(
            "prompt-queue", "prompt-queue-reading"
        )
        self.benchmark_consumer = self.pulsar_client.subscribe(
            "benchmark-prompt-queue", "benchmark-prompt-queue-reading"
        )

        self.producer = self.pulsar_client.create_producer("result-post-discord")
        self.benchmark_producer = self.pulsar_client.create_producer("benchmark-result")

        print("[Txt2Img.init_pulsar]> Pulsar queues loaded")

    def load_model(self):

        model = self.parameters["model"]
        pl_sd = torch.load(model, map_location="cpu")

        if "global_step" in pl_sd:
            print(f"[Txt2Img.load_model]> Model global steps: {pl_sd['global_step']}")

        # Is it better to set self.sd here or to return
        sd = pl_sd["state_dict"]
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

        self.config = OmegaConf.load(f"{self.parameters['config']}")

        self.model = instantiate_from_config(self.config.modelUNet)
        _, _ = self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        self.model.unet_bs = self.parameters["unet_bs"]
        self.model.cdevice = self.parameters["device"]
        self.model.turbo = self.parameters["turbo"]

        self.modelCS = instantiate_from_config(self.config.modelCondStage)
        _, _ = self.modelCS.load_state_dict(sd, strict=False)
        self.modelCS.eval()
        self.modelCS.cond_stage_model.device = self.parameters["device"]

        self.modelFS = instantiate_from_config(self.config.modelFirstStage)
        _, _ = self.modelFS.load_state_dict(sd, strict=False)
        self.modelFS.eval()
        del sd

        if (
            self.parameters["device"] != "cpu"
            and self.parameters["precision"] == "autocast"
        ):
            self.model.half()
            self.modelCS.half()

    def cleanup(self):
        if self.use_pulsar:
            self.pulsar_client.close()
            print("[Txt2Img.cleanup]> Closing pulsar client")

    def pulsar_loop(self):

        print("[Txt2Img.pulsar_loop]> Now waiting for input from pulsar")

        list_of_results = []
        while True:
            benchmark = True
            try:
                msg = self.benchmark_consumer.receive(timeout_millis=1000)
                data = msg.data().decode("utf-8")

                if data == "@@@stop@@@":
                    print(
                        "[Txt2Img.pulsar_loop]> Received end of current benchmark batch"
                    )
                    data = json.dumps(list_of_results)
                    self.benchmark_producer.send(data.encode("utf-8"))
                    list_of_results = []
                else:
                    print(
                        "[Txt2Img.pulsar_loop]> Creating new input from benchmark batch"
                    )
                    config = json.loads(data)
                    result = self.generate_image(config)
                    list_of_results.append(result)

                self.benchmark_consumer.acknowledge(msg)

            except Exception as e:
                benchmark = False

            if not benchmark:
                try:
                    msg = self.consumer.receive(timeout_millis=1000)
                    data = msg.data().decode("utf-8")
                    config = json.loads(data)
                    result = self.generate_image(config)
                    data = json.dumps(result)
                    self.producer.send(data.encode("utf-8"))
                    self.consumer.acknowledge(msg)

                except Exception as e:
                    continue

    def generate_image(self, config):

        config = self.sanitize_config(config)
        print("[Txt2Img.generate_image]> Config sanitized")

        tick = time.time()

        device = self.parameters["device"]

        os.makedirs(config["save_at"], exist_ok=True)
        outpath = config["save_at"]

        if config["seed"] == None:
            config["seed"] = randint(0, 1000000)

        start_code = None
        if config["fixed_code"]:
            start_code = torch.randn(
                [
                    config["samples"],
                    config["latent_channels"],
                    config["height"] // config["downsampling_factors"],
                    config["width"] // config["downsampling_factors"],
                ],
                device=self.parameters["device"],
            )

        batch_size = config["samples"]
        n_rows = config["grid_rows"] if config["grid_rows"] > 0 else batch_size

        prompt = config["prompt"]
        data = [batch_size * [prompt]]

        if self.parameters["precision"] == "autocast" and device != "cpu":
            precision_scope = autocast
        else:
            precision_scope = nullcontext

        with torch.no_grad():
            for n in trange(config["iterations"], desc="Sampling"):
                for prompts in tqdm(data, desc="data"):

                    sample_path = os.path.join(
                        outpath, "_".join(re.split(":| ", prompts[0]))
                    )[:150]
                    os.makedirs(sample_path, exist_ok=True)
                    base_count = len(os.listdir(sample_path))

                    with precision_scope("cuda"):
                        self.modelCS.to(device)
                        uc = None
                        if config["scale"] != 1.0:
                            uc = self.modelCS.get_learned_conditioning(
                                batch_size * [""]
                            )
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
                                    self.modelCS.get_learned_conditioning(
                                        subprompts[i]
                                    ),
                                    alpha=weight,
                                )
                        else:
                            c = self.modelCS.get_learned_conditioning(prompts)

                        shape = [
                            config["samples"],
                            config["latent_channels"],
                            config["height"] // config["downsampling_factors"],
                            config["width"] // config["downsampling_factors"],
                        ]

                        if device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelCS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)

                        samples_ddim = self.model.sample(
                            S=config["steps"],
                            conditioning=c,
                            seed=config["seed"],
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=config["scale"],
                            unconditional_conditioning=uc,
                            eta=config["ddim_eta"],
                            x_T=start_code,
                            sampler=config["sampler"],
                        )

                        self.modelFS.to(device)

                        print(samples_ddim.shape)
                        print("saving images")

                        config["files_path"] = []
                        for i in range(batch_size):

                            x_samples_ddim = self.modelFS.decode_first_stage(
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
                                + str(config["seed"])
                                + "_"
                                + f"{base_count:05}.{config['format']}",
                            )

                            Image.fromarray(x_sample.astype(np.uint8)).save(path)
                            config["files_path"].append(path)
                            base_count += 1
                            config["seed"] += 1

                        if device != "cpu":
                            mem = torch.cuda.memory_allocated() / 1e6
                            self.modelFS.to("cpu")
                            while torch.cuda.memory_allocated() / 1e6 >= mem:
                                time.sleep(1)

                        del samples_ddim
                        print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

        tock = time.time()
        time_taken = (tock - tick) / 60.0

        print(
            (
                "[Txt2Img.generate_image]> Samples finished in {0:.2f} minutes and exported to "
                + sample_path
            ).format(time_taken)
        )

        return config


if __name__ == "__main__":

    load_dotenv()

    parameters = {
        "pulsar_url": os.getenv("PULSAR_URL"),
        "pulsar_token": os.getenv("PULSAR_TOKEN"),
    }

    sd = Txt2Img(parameters, use_pulsar=True)
