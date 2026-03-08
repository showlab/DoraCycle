# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Text-to-image inference with LoRA-loaded DoraCycle checkpoint.
Set config.model.showo.pretrained_model_path to your checkpoint directory
(e.g. experiment.output_dir/checkpoint-XXXX) containing unwrapped_model/ or unwrapped_model_ema/.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import Showo, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    raise ValueError(f"model_type {model_type} not supported.")


def find_linear_layers(model, lora_target_modules):
    """LoRA target: Q/V proj, layers 7-24 (1-based). Exclude layers.0.-.5. (0-indexed)."""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            and all(
                x not in name
                for x in [
                    "vision_tower", "mm_projector",
                    "x_embedder", "t_embedder", "y_embedder", "final_layer",
                ]
            )
            and all(x not in name for x in [f"layers.{i}." for i in range(0, 6)])
            and any(x in name for x in lora_target_modules)
        ):
            lora_module_names.add(name)
    return sorted(list(lora_module_names))


if __name__ == "__main__":
    config = get_config()

    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb.init(
        project="demo",
        name=config.experiment.name + "_t2i_lora",
        config={k: v for k, v in flatten_omega_conf(config, resolve=True).items()},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer,
        max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100,
        cond_dropout_prob=config.training.cond_dropout_prob,
    )

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # LoRA: rank 32, Q/V proj, layers 7-24
    model = Showo(**config.model.showo).to(device)
    lora_target_modules = find_linear_layers(model.showo, "q_proj,v_proj".split(","))
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=lora_target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.showo = get_peft_model(model.showo, lora_config)

    # Load DoraCycle checkpoint: pretrained_model_path = checkpoint dir (e.g. output_dir/checkpoint-XXXX)
    use_ema = getattr(config.model.showo, "use_ema", False)
    subdir = "unwrapped_model_ema" if use_ema else "unwrapped_model"
    ckpt_path = os.path.join(config.model.showo.pretrained_model_path, subdir, "pytorch_model.bin")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Set pretrained_model_path to your DoraCycle checkpoint directory.")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()

    mask_token_id = model.config.mask_token_id
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    batch_size = getattr(config, "batch_size", config.training.batch_size)
    config.training.batch_size = batch_size
    config.training.guidance_scale = getattr(config, "guidance_scale", config.training.guidance_scale)
    config.training.generation_timesteps = getattr(config, "generation_timesteps", config.training.generation_timesteps)

    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    for step in tqdm(range(0, len(validation_prompts), config.training.batch_size)):
        prompts = validation_prompts[step : step + config.training.batch_size]
        image_tokens = torch.ones(
            (len(prompts), config.model.showo.num_vq_tokens),
            dtype=torch.long,
            device=device,
        ) * mask_token_id

        input_ids, _ = uni_prompting((prompts, image_tokens), "t2i_gen")
        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([""] * len(prompts), image_tokens), "t2i_gen")
            attention_mask = create_attention_mask_predict_next(
                torch.cat([input_ids, uncond_input_ids], dim=0),
                pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                rm_pad_in_image=True,
            )
        else:
            attention_mask = create_attention_mask_predict_next(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict["<|pad|>"]),
                soi_id=int(uni_prompting.sptids_dict["<|soi|>"]),
                eoi_id=int(uni_prompting.sptids_dict["<|eoi|>"]),
                rm_pad_in_image=True,
            )
            uncond_input_ids = None

        with torch.no_grad():
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.showo.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        images = (images * 255.0).astype(np.uint8)
        pil_images = [Image.fromarray(img) for img in images]

        wandb_images = [wandb.Image(img, caption=prompts[i]) for i, img in enumerate(pil_images)]
        wandb.log({"generated_images": wandb_images}, step=step)
