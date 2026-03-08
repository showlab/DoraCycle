# coding=utf-8
# Copyright 2024 HuggingFace, NUS Show Lab.
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

import os
import re
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['WANDB_MODE'] = 'disabled'
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.data import Text2ImageDataset
from training.imagenet_dataset import ImageNetDataset
from training.cus_data_parque import CusDataset
from training.openimages_dataset import OpenImagesDataset
from parquet import RefinedWebDataset, create_imagetext_dataloader

from models import Showo, MAGVITv2, VQ_16, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, \
    create_attention_mask_for_mmu, create_attention_mask_for_mmu_perturbed
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from llava.llava_data_vq_unified import get_instruct_data_loader
import torch.nn.functional as F
import deepspeed
import random
import copy

SYSTEM_PROMPT_LEN = 28

from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter, image_transform

from peft import get_peft_model, LoraConfig

try:
    import apex

    is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")

def pcgrad(grad_i, grad_t):
    """Gradient surgery: project grad_t onto orthogonal complement of grad_i (paper Sec 3.3)."""
    dot_product = (grad_i * grad_t).sum()
    if dot_product < 0:
        grad_i_ortho = grad_i - (dot_product / grad_t.norm() ** 2) * grad_t
        grad_all = grad_t + grad_i_ortho
    else:
        grad_all = grad_t + grad_i
    return grad_all


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "vq16":
        return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")
    

def find_first_consecutive_repetition(tensor, min_repeats=3):
    tensor = tensor.flatten()
    diff = tensor[1:] == tensor[:-1]
    diff = diff.to(tensor.device)
    

    changes = torch.cat((torch.tensor([True]).to(tensor.device), diff == False, torch.tensor([True]).to(tensor.device)))
    segment_starts = torch.nonzero(changes).squeeze().to(tensor.device)
    segment_lengths = segment_starts[1:] - segment_starts[:-1]


    for i, length in enumerate(segment_lengths):
        if length >= min_repeats:
            start = segment_starts[i].item()
            return start

    return None 


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with="wandb",
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    total_batch_size_per_gpu = (config.training.batch_size_t2i
                                + config.training.batch_size_lm
                                + config.training.batch_size_mmu)
    total_batch_size = (
            (config.training.batch_size_t2i + config.training.batch_size_lm + config.training.batch_size_mmu)
            * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            total_batch_size_per_gpu
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=config.experiment.name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint")

        accelerator.init_trackers(
            config.experiment.project,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        config_path = Path(config.experiment.output_dir) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Loading models and optimizer")

    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    # unified prompting for show-o
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    print('special tokens : \n', uni_prompting.sptids_dict)

    # VQ model for processing image into discrete tokens
    vq_model = get_vq_model_class(config.model.vq_model.type)
    if config.model.vq_model.get("pretrained_model_path", None):
        vq_model = vq_model().to(accelerator.device)
        state_dict = torch.load(config.model.vq_model.pretrained_model_path, map_location=torch.device('cuda', accelerator.process_index))['model']
        
        vq_model.load_state_dict(state_dict)
    else:
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval()
    vq_model.requires_grad_(False)

    # Initialize Show-o model
    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = config.model.showo.vocab_size - 1
            model.mask_token_id = config.model.showo.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
    mask_id = model.config.mask_token_id

    #################################
    # LoRA: Q/V proj, layers 7-24, rank 32
    ################################

    def find_linear_layers(model, lora_target_modules):
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
                and all(x not in name for x in [f"layers.{i}." for i in range(0, 6)])  # exclude 0-5 -> LoRA on layers 7-24 (1-based)
                and any(x in name for x in lora_target_modules)
            ):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    r = 32
    lora_alpha = 64
    lora_dropout = 0.1
    lora_target_modules = find_linear_layers(model.showo, "q_proj,v_proj".split(","))
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.showo = get_peft_model(model.showo, lora_config)

    ##################################
    #   Optimizer and LR scheduler   #
    #################################
    optimizer_config = config.optimizer.params

    # no decay on bias and layernorm and embedding
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    logger.info("Trainable parameters: ")
    model.showo.print_trainable_parameters()

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    # Create mask scheduler
    if config.get("mask_schedule", None) is not None:
        schedule = config.mask_schedule.schedule
        args = config.mask_schedule.get("params", {})
        mask_schedule = get_mask_chedule(schedule, **args)
    else:
        mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
    )

    ##################################
    #         DATALOADER             #
    #################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_t2i_without_accum = 1 * accelerator.num_processes  #config.training.batch_size_t2i * accelerator.num_processes
    total_batch_size_t2i = (
            1 * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    root_cus = config.experiment.cus_data_path
    characters_names = config.experiment.characters_names

    dataset_cus_img = CusDataset(
        root_cus,
        image_size=preproc_config.resolution,
        special_mark = characters_names[0],
        sample_mode='image',
        sub_ratio=1.0,
        paired_ratio=0.,
    )

    print('process index : ',
            accelerator.process_index, ', ', accelerator.num_processes,
            "Length: ", len(dataset_cus_img))

    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset_cus_img,
                                        num_replicas=accelerator.num_processes,
                                        rank=accelerator.process_index,
                                        shuffle=True,
                                        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_dataloader_t2i_cus = DataLoader(dataset_cus_img, batch_size=config.training.batch_size_t2i,
                                        sampler=sampler, collate_fn=dataset_cus_img.collate_fn,
                                        shuffle=shuffle, num_workers=dataset_config.num_workers)
    num_update_steps_per_epoch = math.ceil(len(dataset_cus_img) / total_batch_size_t2i)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    dataset_cus_txt = CusDataset(
        root_cus,
        image_size=preproc_config.resolution,
        special_mark = characters_names[0],
        sample_mode='text',
        sub_ratio=1.0,
        paired_ratio=0.,
    )


    print('process index : ',
            accelerator.process_index, ', ', accelerator.num_processes,
            "Length: ", len(dataset_cus_txt))

    if accelerator.num_processes > 1:
        sampler = DistributedSampler(dataset_cus_txt,
                                        num_replicas=accelerator.num_processes,
                                        rank=accelerator.process_index,
                                        shuffle=True,
                                        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_dataloader_mmu_cus = DataLoader(dataset_cus_txt, batch_size=config.training.batch_size_mmu,
                                        sampler=sampler, collate_fn=dataset_cus_txt.collate_fn,
                                        shuffle=shuffle, num_workers=dataset_config.num_workers)
    num_update_steps_per_epoch = math.ceil(len(dataset_cus_txt) / total_batch_size_t2i)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    dataset_cus_test = CusDataset(
        root_cus,
        image_size=preproc_config.resolution,
        special_mark = characters_names[0],
        sample_mode='image',
        train_split='test',
        sub_ratio=1.0,
        paired_ratio=1.0,
    )

    sampler = None
    shuffle = True
    train_dataloader_test = DataLoader(dataset_cus_test, batch_size=8,
                                        sampler=sampler, collate_fn=dataset_cus_txt.collate_fn,
                                        shuffle=shuffle, num_workers=dataset_config.num_workers)

    total_batch_size_mmu_without_accum = config.training.batch_size_mmu * accelerator.num_processes

    dataset_lm = RefinedWebDataset(data_path=dataset_config.train_lm_shards_path_or_url,
                                   rank=accelerator.process_index,
                                   world_size=accelerator.num_processes,
                                   num_workers=dataset_config.num_workers)

    train_dataloader_lm = torch.utils.data.DataLoader(dataset_lm, batch_size=config.training.batch_size_lm,
                                                      sampler=None, collate_fn=dataset_lm.collate_fn,
                                                      num_workers=dataset_config.num_workers)

    # Combine these dataloaders into a single iterable model
    iterables = {
        "t2i_flow": train_dataloader_t2i_cus,
        "mmu_flow": train_dataloader_mmu_cus,
        "test_flow": train_dataloader_test,
    }

    combined_dataloader = CombinedLoader(iterables, mode=config.dataset.combined_loader_mode)

    global_step = 0
    first_epoch = 0

    if config.experiment.resume_from_checkpoint:
        dirs = os.listdir(config.experiment.output_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        if path is not None:
            path = os.path.join(config.experiment.output_dir, path)

            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

            accelerator.print(f"Resuming from checkpoint {path}/unwrapped_model/pytorch_model.bin")
            state_dict = torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location=torch.device('cuda', accelerator.process_index))
            model.load_state_dict(state_dict, strict=True)
            del state_dict

    ema_model = copy.deepcopy(model)
    for param in ema_model.showo.parameters():
        param.detach_() 


    first_epoch = global_step // num_update_steps_per_epoch
    logger.info("Preparing model, optimizer and dataloaders")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    vq_model.to(device=accelerator.device)

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.model.embed_tokens.weight.dtype

    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {total_batch_size_per_gpu}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")

    @torch.no_grad()
    def prepare_inputs_and_labels(
            pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
            texts: Union[str, str],
            min_masking_rate: float = 0.0,
            is_train: bool = True,
            is_token_ids: bool = False
    ):
        if not is_token_ids:
            image_tokens = vq_model.get_code(pixel_values_or_image_ids)
            image_tokens = image_tokens + len(uni_prompting.text_tokenizer)
        else:
            image_tokens = pixel_values_or_image_ids

        # create MLM mask and labels
        input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
            image_tokens,
            mask_id,
            config,
            mask_schedule=mask_schedule,
            is_train=is_train,
        )
        input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')

        return input_ids, labels, mask_prob, image_tokens

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    first_epoch = 0
    num_train_epochs = 2000000

    caption_instruction_ids = uni_prompting.text_tokenizer(['USER: \n' + 'Please describe this image in details.' + ' ASSISTANT: '])['input_ids']

    # characters_names = ['Fennec_Fox']
    # characters_names = config.experiment.characters_names
    for epoch in range(first_epoch, num_train_epochs):
        model.train()
        entropy_history = []
        entropy_threshold = 0.8
        for batch, batch_idx, dataloader_idx in combined_dataloader:
            # for loss calculation
            batch_size_t2i = batch["t2i_flow"]["images"].shape[0] #+ batch["mmu_flow_reg"]["images"].shape[0]
            batch_size_lm = 0  #len(batch["lm_flow"]["input_ids"])
            # batch_size_mmu = batch["t2i_flow"]["images"].shape[0]
            batch_size_mmu =len(batch["mmu_flow"]["input_ids"]) #+ len(batch["mmu_flow_reg"]["input_ids"])

            if batch_size_t2i<2:
                batch["t2i_flow"]["images"] = torch.concat([batch["t2i_flow"]["images"] , batch["t2i_flow"]["images"] ], dim=0)
                if batch["t2i_flow"]["input_ids"] is not None:
                    batch["t2i_flow"]["input_ids"] = batch["t2i_flow"]["input_ids"] + batch["t2i_flow"]["input_ids"]
            if batch_size_t2i<2:
                batch["mmu_flow"]["input_ids"] = batch["mmu_flow"]["input_ids"] + batch["mmu_flow"]["input_ids"]

                
                if batch["mmu_flow"]["images"] is not None:
                    batch["mmu_flow"]["images"] = torch.concat([batch["mmu_flow"]["images"] , batch["mmu_flow"]["images"] ], dim=0)

            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            # Build formatted sequences for class-conditional/text-to-image generation
            # *-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*-------*
            pixel_values, texts_indata = batch["t2i_flow"]["images"], batch["t2i_flow"]["input_ids"]

            pixel_values_test, texts_indata_test = batch["test_flow"]["images"], batch["test_flow"]["input_ids"]
            pixel_values_test = pixel_values_test.to(accelerator.device, non_blocking=True)

            if texts_indata is not None and len(texts_indata) == batch_size_t2i:
                cal_I_half_cycle_loss = True
            else:
                cal_I_half_cycle_loss = False

            if texts_indata is not None:
                for idx, prompt in enumerate(texts_indata):
                    if prompt is not None:
                        for name_i in characters_names:
                            name_i_new = "<|sov|> " + name_i + " <|eov|>"
                            prompt = prompt.replace(name_i, name_i_new)
                        texts_indata[idx] = prompt

            pixel_values = pixel_values.to(accelerator.device, non_blocking=True)
            data_time_m.update(time.time() - end)

            pixel_values_mmu, texts_mmu = batch["mmu_flow"]["images"], batch["mmu_flow"]["input_ids"]
            for idx, prompt in enumerate(texts_mmu):
                for name_i in characters_names:
                    name_i_new = "<|sov|> " + name_i + " <|eov|>"
                    prompt = prompt.replace(name_i, name_i_new)
                texts_mmu[idx] = prompt

            if pixel_values_mmu is not None and (pixel_values_mmu.shape[0] == batch_size_mmu):
                cal_T_half_cycle_loss = True
            else:
                cal_T_half_cycle_loss = False

            if global_step == 0 and epoch == 0:
                logger.info("Input ids: {}".format(input_ids))
                logger.info("Labels: {}".format(labels))
            
            with accelerator.accumulate(model): 

                # def scale_gradient(grad):
                #     # print("Original Gradient:", grad)
                #     scaled_grad = grad * 5  
                #     # print("Scaled Gradient:", scaled_grad)
                #     return scaled_grad
                
                
                # T2I_grad_I_cycle = None

                # def save_grad_mid_I_cycle(grad):
                show_mid_results = (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process

                show_mid_results = (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process

                # T cycle: generate pseudo image tokens from text (EMA model)
                if accelerator.mixed_precision == "fp16":
                    weight_dtype = torch.float16
                elif accelerator.mixed_precision == "bf16":
                    weight_dtype = torch.bfloat16
                elif accelerator.mixed_precision == "fp8":
                    weight_dtype = torch.float8
                else:
                    weight_dtype = torch.float32

                with torch.no_grad():
                    ema_model.eval()
                    text_input_T_cycle = texts_mmu
                    mask_token_id = config.model.showo.vocab_size - 1  
                    image_tokens = torch.ones((len(text_input_T_cycle), config.model.showo.num_vq_tokens), dtype=torch.long, device=accelerator.device) * mask_token_id
                    
                    text_ids_T_cycle, input_ids_T_cycle = uni_prompting((text_input_T_cycle, image_tokens), 't2i_gen_cycle')
                    input_ids_T_cycle = input_ids_T_cycle[0]

                    uncond_input_ids, _ = uni_prompting(([''] * len(text_input_T_cycle), image_tokens), 't2i_gen')
                    attention_mask = create_attention_mask_predict_next(torch.cat([input_ids_T_cycle, uncond_input_ids], dim=0), pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']), rm_pad_in_image=True).to(mask_dtype)
                    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                        gen_token_ids = accelerator.unwrap_model(ema_model).t2i_generate(
                            input_ids=input_ids_T_cycle,
                            uncond_input_ids=uncond_input_ids,
                            attention_mask=attention_mask,
                            guidance_scale=5,
                            temperature=config.training.get("generation_temperature", 1.0),
                            timesteps=30,
                            noise_schedule=mask_schedule,
                            noise_type=config.training.get("noise_type", "mask"),
                            predict_all_tokens=config.training.get("predict_all_tokens", False),
                            seq_len=config.model.showo.num_vq_tokens,
                            uni_prompting=uni_prompting,
                            config=config,
                        )
                    image_tokens_T_cycle = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(ema_model).config.codebook_size - 1, min=0)
                    image_tokens_T_cycle = image_tokens_T_cycle.detach()
                    image_tokens_T_cycle = image_tokens_T_cycle + len(uni_prompting.text_tokenizer)

                    if pixel_values_mmu is not None:
                        for idx, pixel_values_mmu_i in enumerate(pixel_values_mmu):
                            if pixel_values_mmu_i is not None:
                                pixel_values_mmu_i = pixel_values_mmu_i.to(accelerator.device, non_blocking=True)
                                pixel_values_mmu_i = vq_model.get_code(pixel_values_mmu_i[None, ])
                                pixel_values_mmu_i = pixel_values_mmu_i + len(uni_prompting.text_tokenizer)
                                image_tokens_T_cycle[idx] = pixel_values_mmu_i

                    if show_mid_results:
                        images = vq_model.decode_code(image_tokens_T_cycle - len(uni_prompting.text_tokenizer))

                        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                        images *= 255.0
                        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                        pil_images_T_cycle = [Image.fromarray(image) for image in images]
                        T_cycle_wandb_images = [wandb.Image(image, caption='T cycle 0 | Inference Image | '+texts_mmu[i]) for i, image in enumerate(pil_images_T_cycle)]

                    # I cycle: generate pseudo text from image (EMA model)
                    image_tokens_I_cycle = vq_model.get_code(pixel_values)
                    image_tokens_I_cycle = image_tokens_I_cycle + len(uni_prompting.text_tokenizer)


                    if show_mid_results:
                        images = vq_model.decode_code(image_tokens_I_cycle - len(uni_prompting.text_tokenizer))
                        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                        images *= 255.0
                        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                        pil_images_I_cycle = [Image.fromarray(image) for image in images]

                    texts_I_cycle = []
                    I_cycle_wandb_images = []
                    global_entropy_all = []
                    for idx, image_tokens_I_cycle_i in enumerate(image_tokens_I_cycle):
                        if texts_indata[idx] is None:
                        # if texts_indata is None:
                            image_tokens_I_cycle_i = image_tokens_I_cycle_i[None, ]

                            caption_instruction_ids_ten = torch.tensor(caption_instruction_ids).to(accelerator.device)
                            input_id_I_cycle = torch.cat([
                                (torch.ones(caption_instruction_ids_ten.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(accelerator.device),
                                (torch.ones(caption_instruction_ids_ten.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(accelerator.device),
                                image_tokens_I_cycle_i,
                                (torch.ones(caption_instruction_ids_ten.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(accelerator.device),
                                (torch.ones(caption_instruction_ids_ten.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(accelerator.device),
                                caption_instruction_ids_ten
                            ], dim=1).long()
                            
                                
                            attention_mask_I_cycle, attention_mask_I_cycle_perturbed = create_attention_mask_for_mmu_perturbed(input_id_I_cycle.to(accelerator.device),
                                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
                            attention_mask_I_cycle = attention_mask_I_cycle.to(mask_dtype)
                            attention_mask_I_cycle_perturbed = attention_mask_I_cycle_perturbed.to(mask_dtype)

                            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                                result = accelerator.unwrap_model(ema_model).mmu_generate(
                                    input_id_I_cycle,
                                    attention_mask=attention_mask_I_cycle,
                                    max_new_tokens=config.dataset.preprocessing.max_seq_length,
                                    top_k=1,
                                    temperature=1.0,
                                    eot_token=uni_prompting.sptids_dict['<|eot|>'],
                                )
                            result = torch.stack(result).squeeze()[None]

                            
                            end_index = find_first_consecutive_repetition(result)
                            if end_index is not None:
                                result = result[:, :end_index]
                            text = uni_prompting.text_tokenizer.batch_decode(result, skip_special_tokens=True)
                            text[0] = text[0].replace("black and white ", " ")
                            if characters_names[0] not in text[0]:
                                text = ["<|sov|> " + characters_names[0] + " <|eov|> " + text[0]]
                        else:
                            text = [texts_indata[idx]]

                        if show_mid_results:
                            I_cycle_wandb_images.append(wandb.Image(pil_images_I_cycle[idx], caption='I cycle 0 | Inference Text | '+ text[0].replace('\n', '')))
                        text = 'USER: \n' + 'Please describe this image in details.' + ' ASSISTANT:' + ' ' + ' ' + text[0].replace('\n', '')
                        texts_I_cycle.append(text)
                model.train()

                # # <<<<<<<<<<<<  T cycle 02: input text and generate image tokens >>>>>>>>>>>>>

                # create MLM mask and labels
                # input_ids_T_cycle, labels_T_cycle, loss_weight_T_cycle, mask_prob_T_cycle = mask_or_random_replace_tokens(
                #     image_tokens_T_cycle,
                #     mask_id,
                #     config,
                #     mask_schedule=mask_schedule,
                #     is_train=True,
                # )

                # input_ids_T_cycle, masks_T_cycle, labels_T_cycle = uni_prompting((text_input_T_cycle, input_ids_T_cycle, labels_T_cycle), 't2i_predict_next')
                
                (
                    input_ids_T_cycle,
                    labels_T_cycle,
                    mask_prob,
                    image_tokens_ori
                ) = prepare_inputs_and_labels(image_tokens_T_cycle, text_input_T_cycle, config.training.min_masking_rate, is_token_ids=True)

                attention_mask_T_cycle = create_attention_mask_predict_next(input_ids_T_cycle, pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']), rm_pad_in_image=True, return_inverse_mask=True)
                attention_mask_T_cycle = attention_mask_T_cycle.to(mask_dtype)


                # <<<<<<<<<<<<<<<<<<<<<<<< I cycle 02: generate text tokens >>>>>>>>>>>>>>>>>>>>>
                input_ids_I_cycle, _, labels_I_cycle = uni_prompting((image_tokens_I_cycle, texts_I_cycle), 'mmu')
                # print(input_ids_I_cycle[0, (config.model.showo.num_vq_tokens + 4 + 15):])
                # print("\n input text \n")
                # print(uni_prompting.text_tokenizer.batch_decode(input_ids_I_cycle[0, (config.model.showo.num_vq_tokens + 4 + 15):].unsqueeze(0), skip_special_tokens=True)[0])
                
                attention_mask_I_cycle = create_attention_mask_for_mmu(input_ids_I_cycle.to(accelerator.device),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>'])).to(mask_dtype)

                # <<<<<<<<<<<<<<<<<<<<<<<<<< First half cycle >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                attention_mask_cycle_1 = torch.cat([attention_mask_T_cycle, attention_mask_I_cycle], dim=0)
                input_ids_cycle_1 = torch.cat((input_ids_T_cycle, input_ids_I_cycle.to(accelerator.device)), dim=0) 

                
                    
                # for param in model.parameters():
                #     param.register_hook(lambda grad: save_grad_1(grad)) 

                with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                    logits_cycle_1 = model(
                        input_ids=input_ids_cycle_1,
                        attention_mask=attention_mask_cycle_1,
                        labels=None,
                    )        

                loss = 0
                if cal_T_half_cycle_loss:
                    loss_t2i_half_cycle = F.cross_entropy(
                        logits_cycle_1[:batch_size_mmu, config.dataset.preprocessing.max_seq_length + 1:].contiguous().view(-1, config.model.showo.vocab_size),
                        labels_T_cycle[:batch_size_mmu, config.dataset.preprocessing.max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                    )
                    loss += loss_t2i_half_cycle
                
                if cal_I_half_cycle_loss:

                    loss_mmu_half_cycle = F.cross_entropy(
                        logits_cycle_1[-batch_size_t2i:, :-1].contiguous().view(-1, config.model.showo.vocab_size),
                        labels_I_cycle[-batch_size_t2i:, 1:].contiguous().view(-1), ignore_index=-100,
                    )
                    loss += loss_mmu_half_cycle

                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         param.register_hook(gradient_surgery(name))

                # <<<<<<<<<< T cycle 03: generate text tokens and cal loss on text >>>>>>>>>>>>>>
                t2i_predicted_i = logits_cycle_1[:config.training.batch_size_mmu, config.dataset.preprocessing.max_seq_length+1+1:-1]
                t2i_predicted_i.requires_grad_()
                t2i_predicted_i.retain_grad() 
                t2i_predicted_i_gs = torch.nn.functional.gumbel_softmax(t2i_predicted_i, tau=1.0, hard=True)

                T_cycle_1_gen_imgs = []
                if show_mid_results:

                    images = vq_model.decode_code(t2i_predicted_i.argmax(dim=-1) - len(uni_prompting.text_tokenizer))

                    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                    images *= 255.0
                    try:
                        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                        T_cycle_1_gen_imgs = [Image.fromarray(image) for image in images]
                        
                        T_cycle_wandb_images += [wandb.Image(image, caption='T cycle 1 | Generated Image | ' + texts_mmu[i]) for i, image in enumerate(T_cycle_1_gen_imgs)]
                    
                    except RuntimeError as e:
                        show_mid_results = False

                text_ids_T_cycle_with_instruction = []
                for text_ids_T_cycle_i in text_ids_T_cycle:

                    text_ids_T_cycle_i = caption_instruction_ids[0] + text_ids_T_cycle_i[1:]  # remove eot token
                    text_ids_T_cycle_with_instruction.append(text_ids_T_cycle_i)
                    
                    # print("T2I2T Cycle 2 Input Text")
                    # text = uni_prompting.text_tokenizer.batch_decode(torch.tensor(text_ids_T_cycle_i).unsqueeze(0))[0]
                    # print(text)

                input_ids_T_cycle_2, mask_T_cycle_2, labels_T_cycle_2 = uni_prompting.mmu_prompt_one_hot(t2i_predicted_i_gs, text_ids_T_cycle_with_instruction)

                # for input_ids_T_cycle_2_i in input_ids_T_cycle_2:
                    
                #     print("input_ids_T_cycle_2_i")
                #     text = uni_prompting.text_tokenizer.batch_decode(input_ids_T_cycle_2_i.argmax(dim=-1).unsqueeze(0)[:, (config.model.showo.num_vq_tokens + 4 + 15):])[0]
                #     print(text)
                
                # print(f"shape of input_ids_T_cycle_2 {input_ids_T_cycle_2.shape}")
                # print(f"shape of mask_T_cycle_2 {mask_T_cycle_2.shape}")
                # print(f"shape of labels_T_cycle_2 {labels_T_cycle_2.shape}")

                # input_ids_cap = input_ids_cap.to(input_ids.device)
                attention_mask_T_cycle_2 = create_attention_mask_for_mmu(input_ids_T_cycle_2.argmax(-1).to(accelerator.device),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>'])).to(mask_dtype)
                  

                # <<<<<<<<<< I cycle 03: generate image tokens and cal loss on image >>>>>>>>>>>>>>
                
                # i2t_predicted_t = logits_cycle_1[-config.training.batch_size:, (config.model.showo.num_vq_tokens + 3):-1] 
                
                # logits_cycle_1.retain_grad() 

                i2t_predicted_t = logits_cycle_1[-config.training.batch_size_t2i:, (config.model.showo.num_vq_tokens + 4 + 15 -1):] 
                # i2t_predicted_t.register_hook(orthogonalize_grad_I_cycle)
                # i2t_predicted_t.register_hook(scale_gradient)
                i2t_predicted_t.requires_grad_()
                i2t_predicted_t.retain_grad() 

                i2t_predicted_t_gs = torch.nn.functional.gumbel_softmax(i2t_predicted_t, tau=1.0, hard=True)

                i2t_predicted_texts = []
                texts_I_cycle_1 = []
                for idx, i2t_predicted_t_sub in enumerate(i2t_predicted_t_gs):
                    # print(i2t_predicted_t_sub)
                    # start_index = torch.where(i2t_predicted_t_sub == 25)[0][1]
                    i2t_predicted_t_sub_int = i2t_predicted_t_sub.argmax(dim=-1)
                    
                    # print("\n GENERATED  + 4 + 15 -1 - 1 text \n")
                    # print(uni_prompting.text_tokenizer.batch_decode(logits_cycle_1[-2, (config.model.showo.num_vq_tokens + 4):, :].argmax(dim=-1).unsqueeze(0), skip_special_tokens=True)[0])
                    
                    # print("\n GENERATED  + 4 + 15 -1 text \n")
                    # print(uni_prompting.text_tokenizer.batch_decode(logits_cycle_1[-2, (config.model.showo.num_vq_tokens + 4 + 15 -1):, ].argmax(dim=-1).unsqueeze(0), skip_special_tokens=True)[0])
                    # start_index = 15
                    try:
                        end_index_1 = torch.where(i2t_predicted_t_sub_int == 50256)[0][0] 
                        end_index_2 = find_first_consecutive_repetition(i2t_predicted_t_sub_int)
                        if end_index_2 is not None:
                            end_index = min(end_index_1, end_index_2)
                        else:
                            end_index = end_index_1
                    except IndexError:
                        end_index = find_first_consecutive_repetition(i2t_predicted_t_sub_int)
                    # if start_index is None:
                    #     start_index = 0
                    if end_index is not None and end_index > 0:
                        # end_index = -1
                        i2t_predicted_t_sub_puretext = i2t_predicted_t_sub[:end_index, ]
                    else:
                        i2t_predicted_t_sub_puretext = i2t_predicted_t_sub

                    # print(f"end_index: {end_index}")
                    
                    # i2t_predicted_t_sub_puretext.requires_grad_() 
                    # i2t_predicted_t_sub_puretext.retain_grad()

                    # # print (i2t_predicted_t_sub_int)
                    # text = uni_prompting.text_tokenizer.batch_decode(i2t_predicted_t_sub_int.unsqueeze(0))[0]
                    # print(text)
                    
                    if show_mid_results:
                        text = uni_prompting.text_tokenizer.batch_decode(i2t_predicted_t_sub_int[:end_index].unsqueeze(0), skip_special_tokens=True)[0]
                        texts_I_cycle_1.append(text)
                        I_cycle_wandb_images.append(wandb.Image(pil_images_I_cycle[idx], caption='I cycle 1 | Generated Text | '+ text))
                    # print("I2T2I Predicted Text")
                    # print(text)
                    # print(f"shape of i2t_predicted_t_sub_puretext {i2t_predicted_t_sub_puretext.shape}")

                    i2t_predicted_texts.append(i2t_predicted_t_sub_puretext)
                    # i2t_predicted_texts.append(text)               

                # create MLM mask and labels
                image_tokens_I_cycle_2 = image_tokens_I_cycle.clone()   
                

                # ###################### CHECK IMAGE ##############################
                # images = vq_model.decode_code(image_tokens_I_cycle - len(uni_prompting.text_tokenizer))

                # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                # images *= 255.0
                # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                # pil_images = [Image.fromarray(image) for image in images]
                # for idx, img in enumerate(pil_images):
                #     img.save(f'Check Image I Cycle Input image_{idx}.png')
                # ##################################################################


                input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
                    image_tokens_I_cycle_2,
                    mask_id,
                    config,
                    mask_schedule=mask_schedule,
                    is_train=True,
                )


                input_ids_I_cycle_2, masks_I_cycle_2, labels_I_cycle_2 = uni_prompting.t2i_prompt_predict_next_tensor_one_hot(i2t_predicted_texts, input_ids, labels)
                # print(f"shape of input_ids_I_cycle_2 {input_ids_I_cycle_2.shape}")
                # print(f"shape of masks_I_cycle_2 {masks_I_cycle_2.shape}")
                # print(f"shape of labels_I_cycle_2 {labels_I_cycle_2.shape}")

                # input_ids_I_cycle_2, masks_I_cycle_2, labels_I_cycle_2 = uni_prompting((i2t_predicted_texts, input_ids, labels), 't2i_predict_next')

                attention_mask_I_cycle_2 = create_attention_mask_predict_next(input_ids_I_cycle_2.argmax(-1), pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                    # eoi_id=int(uni_prompting.sptids_dict['eoi']), rm_pad_in_image=True, return_inverse_mask=False)
                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']), rm_pad_in_image=True, return_inverse_mask=True)
                attention_mask_I_cycle_2 = attention_mask_I_cycle_2.to(mask_dtype)
                # attention_mask_cycle_2 = torch.cat([attention_mask_T_cycle_2, attention_mask_I_cycle_2], dim=0)
                # input_ids_cycle_2 = torch.cat((input_ids_T_cycle_2, input_ids_I_cycle_2.to(input_ids_I_cycle.device)), dim=0)
                # labels_cycle_2 = torch.cat((labels_T_cycle_2, labels_I_cycle_2.to(input_ids_I_cycle.device)), dim=0)

                # attention_mask_cycle_2 = attention_mask_I_cycle_2
                # input_ids_cycle_2 = input_ids_I_cycle_2.to(input_ids_I_cycle.device)
                # labels_cycle_2 = labels_I_cycle_2.to(input_ids_I_cycle.device)

                # input_ids_cycle_2.retain_grad()
                # <<<<<<<<<<<<<<<<<<<<<<<<<< Second half cycle >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                # attention_mask_cycle_2 = torch.cat([attention_mask_T_cycle_2, attention_mask_I_cycle_2], dim=0)
                # input_ids_cycle_2 = torch.cat((input_ids_T_cycle_2, input_ids_I_cycle_2.to(accelerator.device)), dim=0)
                # labels_cycle_2 = torch.cat((labels_T_cycle_2, labels_I_cycle_2.to(accelerator.device)), dim=0)

                attention_mask_cycle_2 = torch.cat([attention_mask_I_cycle_2, attention_mask_T_cycle_2], dim=0)
                input_ids_cycle_2 = torch.cat((input_ids_I_cycle_2.to(accelerator.device), input_ids_T_cycle_2), dim=0) 
                labels_cycle_2 = torch.cat((labels_I_cycle_2.to(accelerator.device), labels_T_cycle_2), dim=0) 

                with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):

                    # logits_cycle_2, loss_t2i, loss_lm, loss_mmu = model(
                        
                    logits_cycle_2 = model(
                        input_ids=input_ids_cycle_2,
                        input_embeddings=None,
                        attention_mask=attention_mask_cycle_2,
                        labels=None,
                        label_smoothing=config.training.label_smoothing,
                        batch_size_t2i=batch_size_t2i,
                        batch_size_lm=batch_size_lm,
                        batch_size_mmu=batch_size_mmu,
                        max_seq_length=config.dataset.preprocessing.max_seq_length,
                        one_hot=True,
                    )

                # logits_cycle_2.register_hook(save_grad_mid_I_cycle)
                logits_I_cycle_2 = logits_cycle_2[:batch_size_t2i, config.dataset.preprocessing.max_seq_length + 1:]

                loss_t2i = F.cross_entropy(
                    # logits_cycle_2[:batch_size_t2i, config.dataset.preprocessing.max_seq_length + 1:].contiguous().view(-1, model.output_size),
                    logits_I_cycle_2.contiguous().view(-1, config.model.showo.vocab_size),
                    labels_cycle_2[:batch_size_t2i, config.dataset.preprocessing.max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )

                loss_mmu = F.cross_entropy(
                    logits_cycle_2[-batch_size_mmu:, :-1].contiguous().view(-1, config.model.showo.vocab_size),
                    labels_cycle_2[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss_t2i = accelerator.gather(loss_t2i.repeat(config.training.batch_size_t2i)).mean()
                # avg_loss_lm = accelerator.gather(loss_lm.repeat(config.training.batch_size_lm)).mean()
                avg_loss_mmu = accelerator.gather(loss_mmu.repeat(config.training.batch_size_mmu)).mean()
                # loss = config.training.t2i_coeff * loss_t2i + \
                #     #    config.training.lm_coeff * loss_lm + \
                #        config.training.mmu_coeff * loss_mmu

                
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         param.register_hook(save_good_gradient(name))

                avg_masking_rate = accelerator.gather(mask_prob.repeat(config.training.batch_size_t2i)).mean()

                grad_p = None
                if cal_T_half_cycle_loss or cal_I_half_cycle_loss:
                    accelerator.backward(loss, retain_graph=True)
                    grad_p = []
                    for param in model.parameters():
                        if param.grad is not None:
                            grad_p.append(param.grad.clone())

                accelerator.backward(loss_t2i, retain_graph=True)
                grad_i = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_i.append(param.grad.clone())


                optimizer.zero_grad(set_to_none=True)

                beta_t_cycle = config.training.get("t_cycle_coeff", 0.1)
                loss_mmu = beta_t_cycle * loss_mmu
                accelerator.backward(loss_mmu)
                grad_t = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_t.append(param.grad.clone())

                adjusted_grads = [pcgrad(gt, gi) for gt, gi in zip(grad_t, grad_i)]

                for param, adj_grad in zip([param for param in model.parameters() if param.grad is not None], adjusted_grads):
                    param.grad = adj_grad 
                
                if grad_p is not None:
                    for param, adj_grad in zip([param for param in model.parameters() if param.grad is not None], grad_p):
                        param.grad += adj_grad 


                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                
                ema_alpha = config.training.get("ema_decay", 0.999)
                def update_ema(model, ema_model, alpha):
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
                update_ema(model, ema_model, alpha=ema_alpha)

                # log gradient norm before zeroing it
                if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                batch_time_m.update(time.time() - end)
                end = time.time()

                # Log metrics
                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                            config.training.gradient_accumulation_steps * total_batch_size_per_gpu / batch_time_m.val
                    )
                    logs = {
                        "I_Cycle_Loss": avg_loss_t2i.item(),
                        "T_Cycle_Loss": avg_loss_mmu.item(),
                        # "average_entropy": sum(global_entropy_all)/len(global_entropy_all),
                        # "entropy_threshold": entropy_threshold,
                        # "step_loss_lm": avg_loss_lm.item(),
                        # "gradient_T_cycle": grads_T.item(),
                        # "gradient_I_cycle": grads_I.item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "avg_masking_rate": avg_masking_rate.item(),
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "data_time": data_time_m.val,
                        "batch_time": batch_time_m.val,
                    }
                    accelerator.log(logs, step=global_step + 1)

                    logger.info(
                        f"Step: {global_step + 1} "
                        f"Loss_t2i: {avg_loss_t2i.item():0.4f} "
                        f"Loss_mmu: {avg_loss_mmu.item():0.4f} "
                        # f"Loss_lm: {avg_loss_lm.item():0.4f} "
                        f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f}"
                    )

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, ema_model, config, accelerator, global_step + 1)

                # if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                if show_mid_results:

                    t2i_predicted_i = logits_cycle_2[:config.training.batch_size_t2i, config.dataset.preprocessing.max_seq_length+1+1:-1]
                    images = vq_model.decode_code(t2i_predicted_i.argmax(dim=-1) - len(uni_prompting.text_tokenizer))

                    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                    images *= 255.0
                    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
                    pil_images = [Image.fromarray(image) for image in images]
                    
                    I_cycle_wandb_images += [wandb.Image(image, caption='I cycle 2 | Generated Image | ' + texts_I_cycle_1[i]) for i, image in enumerate(pil_images)]


                    i2t_predicted_t = logits_cycle_2[-config.training.batch_size_mmu:, (config.model.showo.num_vq_tokens + 4 + 15 -1):] 
                                        
                    i2t_predicted_texts = []
                    for i2t_predicted_t_sub in i2t_predicted_t:
                        i2t_predicted_t_sub_int = i2t_predicted_t_sub.argmax(dim=-1)
                        try:
                            end_index_1 = torch.where(i2t_predicted_t_sub_int == 50256)[0][0] 
                            end_index_2 = find_first_consecutive_repetition(i2t_predicted_t_sub_int)
                            if end_index_2 is not None:
                                end_index = min(end_index_1, end_index_2)
                            else:
                                end_index = end_index_1
                        except IndexError:
                            end_index = find_first_consecutive_repetition(i2t_predicted_t_sub_int)
                        if end_index is not None and end_index > 0:
                            i2t_predicted_t_sub_puretext = i2t_predicted_t_sub[:end_index, ]
                        else:
                            i2t_predicted_t_sub_puretext = i2t_predicted_t_sub

                        text = uni_prompting.text_tokenizer.batch_decode(i2t_predicted_t_sub_puretext.argmax(dim=-1).unsqueeze(0))[0]

                        

                        i2t_predicted_texts.append(text)

                    T_cycle_wandb_images+= [wandb.Image(image, caption='T cycle 2 | Generated Text | ' + i2t_predicted_texts[i]) for i, image in enumerate(T_cycle_1_gen_imgs)]


                    wandb.log({"I_cycle Vis": I_cycle_wandb_images}, step=global_step+1)
                    wandb.log({"T_cycle Vis": T_cycle_wandb_images}, step=global_step+1)


                    # pixel_values_test, texts_indata_test

                    visualize_cycles(       
                        pixel_values_test,         
                        model,  #ema_model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        weight_dtype,
                        mask_schedule=mask_schedule,
                    )

                    # visualize_cycles_train(                
                    #     ema_model,
                    #     pixel_values,
                    #     vq_model,
                    #     uni_prompting,
                    #     accelerator,
                    #     config,
                    #     global_step + 1,
                    #     weight_dtype,
                    #     mask_schedule=mask_schedule,
                    # )

                    if hasattr(model, 'module'):
                        mask_dtype = model.module.showo.model.model.embed_tokens.weight.dtype
                    else:
                        mask_dtype = model.showo.model.embed_tokens.weight.dtype

                    generate_images(
                        texts_indata_test,
                        model,  #ema_model,
                        vq_model,
                        uni_prompting,
                        accelerator,
                        config,
                        global_step + 1,
                        mask_schedule,
                        mask_dtype,
                        characters_names
                    )

                    # generate_images(
                    #     model,
                    #     vq_model,
                    #     uni_prompting,
                    #     accelerator,
                    #     config,
                    #     global_step + 1,
                    #     mask_schedule=mask_schedule,
                    # )

                    # visualize_predictions(
                    #     model,
                    #     vq_model,
                    #     uni_prompting,
                    #     config,
                    #     global_step + 1,
                    #     input_ids,
                    #     image_tokens_ori,
                    #     batch["t2i_flow"]["images"],
                    #     texts,
                    #     logits,
                    # )

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                break
            # End for

    accelerator.wait_for_everyone()

    # Evaluate and save checkpoint at the end of training
    save_checkpoint(model, ema_model, config, accelerator, global_step)

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(config.experiment.output_dir, safe_serialization=False)

    accelerator.end_training()


@torch.no_grad()
def visualize_cycles(
        test_images,
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        weight_dtype,
        mask_schedule,
):
    logger.info("Visualizing Cycles...")
    model.eval()
    cycle_test_imgs_root = os.path.join(config.experiment.cus_data_path, 'cycle_test_imgs')

    images = []
    question = 'Please describe this image in details.'
    device = accelerator.device
    # wandb_images = []
    validation_prompts = []
    # config.question = config.question.split(' *** ')
    # for i, file_name in enumerate(os.listdir(cycle_test_imgs_root)):
    #     image_path = os.path.join(cycle_test_imgs_root, file_name)
    #     image_ori = Image.open(image_path).convert("RGB")
    #     image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
    #     image = image.unsqueeze(0)
    if test_images is not None:
        for i, image in enumerate(test_images):
            image = image.unsqueeze(0)
            images.append(image)        
            
            image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
            batch_size = 1

            input_ids = uni_prompting.text_tokenizer(['USER: \n' + question + ' ASSISTANT:'])['input_ids']
            input_ids = torch.tensor(input_ids).to(device)

            input_ids = torch.cat([
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
                image_tokens,
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
                (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
                input_ids
            ], dim=1).long()

            attention_mask = create_attention_mask_for_mmu(input_ids.to(device),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                cont_toks_list = accelerator.unwrap_model(model).mmu_generate(input_ids, attention_mask=attention_mask,
                        max_new_tokens=config.dataset.preprocessing.max_seq_length, top_k=1, temperature=1.0,
                        eot_token=uni_prompting.sptids_dict['<|eot|>'])
            
                cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
                text_generated = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0]
                
                validation_prompts.append(text_generated)
            
        images = torch.cat(images, dim=0)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]

            


        # read validation prompts from file
        # with open(config.dataset.params.validation_prompts_file, "r") as f:
        #     validation_prompts = f.read().splitlines()[:24]
        guidance_scale=5
        generation_timesteps = 50
        if hasattr(model, 'module'):
            mask_dtype = model.module.showo.model.model.embed_tokens.weight.dtype
        else:
            mask_dtype = model.showo.model.model.embed_tokens.weight.dtype

        mask_token_id = config.model.showo.vocab_size - 1
        image_tokens = torch.ones((len(validation_prompts), config.model.showo.num_vq_tokens), dtype=torch.long,
                                device=accelerator.device) * mask_token_id
        input_ids, _ = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
        if guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True).to(mask_dtype)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True).to(mask_dtype)
            uncond_input_ids = None

        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            # Generate images
            gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                predict_all_tokens=config.training.get("predict_all_tokens", False),
                seq_len=config.model.showo.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )
        # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
        # so we clamp them to the correct range.
        gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)

        model.train()

        if config.training.get("pre_encode", False):
            del vq_model

        # Convert to PIL images
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        # Log images
        wandb_images += [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
        wandb.log({"Cycle Test": wandb_images}, step=global_step)


@torch.no_grad()
def visualize_cycles_train(
        model,
        pixel_values,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        weight_dtype,
        mask_schedule,
):
    logger.info("Visualizing Cycles...")
    model.eval()
    cycle_test_imgs_root = os.path.join(config.experiment.cus_data_path, 'cycle_test') if hasattr(config.experiment, 'cus_data_path') else './cycle_test'

    images = []
    question = 'Please describe this image in details.'
    device = accelerator.device
    # wandb_images = []
    validation_prompts = []
    # config.question = config.question.split(' *** ')
    for i in range(pixel_values.shape[0]):
        # image_path = os.path.join(cycle_test_imgs_root, file_name)
        # image_ori = Image.open(image_path).convert("RGB")
        # image = image_transform(image_ori, resolution=config.dataset.params.resolution).to(device)
        # image = image.unsqueeze(0)
        image = pixel_values[i].unsqueeze(0)
        images.append(image)        
        
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        batch_size = 1

        input_ids = uni_prompting.text_tokenizer(['USER: \n' + question + ' ASSISTANT:'])['input_ids']
        input_ids = torch.tensor(input_ids).to(device)

        input_ids = torch.cat([
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|mmu|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|soi|>']).to(device),
            image_tokens,
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|eoi|>']).to(device),
            (torch.ones(input_ids.shape[0], 1) * uni_prompting.sptids_dict['<|sot|>']).to(device),
            input_ids
        ], dim=1).long()

        attention_mask = create_attention_mask_for_mmu(input_ids.to(device),
                                                        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']))
        with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
            cont_toks_list = accelerator.unwrap_model(model).mmu_generate(input_ids, attention_mask=attention_mask,
                    max_new_tokens=config.dataset.preprocessing.max_seq_length, top_k=1, temperature=1.0,
                    eot_token=uni_prompting.sptids_dict['<|eot|>'])
        
            cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]
            text_generated = uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0]
            
            validation_prompts.append(text_generated)
        
    images = torch.cat(images, dim=0)
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]

        


    # read validation prompts from file
    # with open(config.dataset.params.validation_prompts_file, "r") as f:
    #     validation_prompts = f.read().splitlines()[:24]
    guidance_scale=5
    generation_timesteps = 50
    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.model.embed_tokens.weight.dtype

    mask_token_id = config.model.showo.vocab_size - 1
    image_tokens = torch.ones((len(validation_prompts), config.model.showo.num_vq_tokens), dtype=torch.long,
                            device=accelerator.device) * mask_token_id
    input_ids, _ = uni_prompting((validation_prompts, image_tokens), 't2i_gen')
    if guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True).to(mask_dtype)
    else:
        attention_mask = create_attention_mask_predict_next(input_ids,
                                                            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                            rm_pad_in_image=True).to(mask_dtype)
        uncond_input_ids = None

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
        # Generate images
        gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
            input_ids=input_ids,
            uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask,
            guidance_scale=guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=generation_timesteps,
            noise_schedule=mask_schedule,
            noise_type=config.training.get("noise_type", "mask"),
            predict_all_tokens=config.training.get("predict_all_tokens", False),
            seq_len=config.model.showo.num_vq_tokens,
            uni_prompting=uni_prompting,
            config=config,
        )
    # In the beginning of training, the model is not fully trained and the generated token ids can be out of range
    # so we clamp them to the correct range.
    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)

    model.train()

    if config.training.get("pre_encode", False):
        del vq_model

    # Convert to PIL images
    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(image) for image in images]

    # Log images
    wandb_images += [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
    wandb.log({"Cycle Test Training Data": wandb_images}, step=global_step)



@torch.no_grad()
def visualize_predictions(
        model,
        vq_model,
        uni_prompting,
        config,
        global_step,
        input_ids,
        image_tokens_ori,
        ori_images,
        texts,
        logits,
):
    logger.info("Visualizing predictions...")
    model.eval()

    recons_images = vq_model.decode_code(image_tokens_ori - len(uni_prompting.text_tokenizer))
    recons_images = torch.clamp((recons_images + 1.0) / 2.0, min=0.0, max=1.0)
    recons_images *= 255.0
    recons_images = recons_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    images = torch.clamp((ori_images + 1.0) / 2.0, min=0.0, max=1.0)
    images *= 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

    predictions = logits[:config.training.batch_size_t2i, -(config.model.showo.num_vq_tokens + 1):-1:,
                  config.model.showo.llm_vocab_size + config.model.showo.num_new_special_tokens:-1]
    predictions = predictions.argmax(axis=-1)

    mask_token_id = config.model.showo.vocab_size - 1 - len(uni_prompting.text_tokenizer)
    input_ids = input_ids[:config.training.batch_size_t2i, -(config.model.showo.num_vq_tokens + 1):-1:] - len(
        uni_prompting.text_tokenizer)
    mask_ratio = list((torch.where(input_ids == mask_token_id, 1, 0).sum(
        dim=-1) / config.model.showo.num_vq_tokens).cpu().numpy())
    predicted_images = torch.where(input_ids == mask_token_id, predictions, input_ids)

    predicted_images = vq_model.decode_code(predicted_images)
    predicted_images = torch.clamp((predicted_images + 1.0) / 2.0, min=0.0, max=1.0)
    predicted_images *= 255.0
    predicted_images = predicted_images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    predicted_images = np.concatenate((images, recons_images, predicted_images), 2)
    pil_images = [Image.fromarray(image) for image in predicted_images]

    # Log images
    wandb_images = [wandb.Image(image, caption=f'mask ratio: {r:0.2f} \n caption: {texts[i]}') for i, (image, r) in
                    enumerate(zip(pil_images, mask_ratio))]
    wandb.log({"Original images v.s. Reconstructed images v.s. Predicted images": wandb_images}, step=global_step)

    model.train()


@torch.no_grad()
def generate_images(
        test_text,
        model,
        vq_model,
        uni_prompting,
        accelerator,
        config,
        global_step,
        mask_schedule,
        mask_dtype,
        characters_names,
):
    logger.info("Generating images...")
    model.eval()

    # read validation prompts from file
    val_prompt_path = os.path.join(config.experiment.cus_data_path, 'validation_prompts.txt')
    # with open(config.dataset.params.validation_prompts_file, "r") as f:
    with open(val_prompt_path, "r") as f:
        validation_prompts = f.read().splitlines()[:24]

    # validation_prompts = test_text
    if len(validation_prompts) > 0:
        
        
        for idx, prompt in enumerate(validation_prompts):
            for name_i in characters_names:  #['Nobita', 'Doraemon', 'Gian', 'Suneo', 'Shizuka', 'White cat', 'white cat']:
                name_i_new = "<|sov|> "+name_i+" <|eov|>"
                prompt = prompt.replace(name_i, name_i_new)
            # validation_prompts[idx] = "In the customized comic style. " + prompt
            validation_prompts[idx] = prompt

        # if hasattr(model, 'module'):
        #     mask_dtype = model.module.showo.model.model.embed_tokens.weight.dtype
        # else:
        #     mask_dtype = model.showo.model.embed_tokens.weight.dtype

        guidance_scale=5
        generation_timesteps = 50

        mask_token_id = config.model.showo.vocab_size - 1
        batch_size = len(validation_prompts) // 2

        pil_images = []

        for batch_idx in range(2):
            batch_prompts = validation_prompts[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            # print(batch_prompts)
            image_tokens = torch.ones((len(batch_prompts), config.model.showo.num_vq_tokens), dtype=torch.long,
                                    device=accelerator.device) * mask_token_id
            input_ids, _ = uni_prompting((batch_prompts, image_tokens), 't2i_gen')

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(batch_prompts), image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True).to(mask_dtype)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True).to(mask_dtype)
                uncond_input_ids = None

            if accelerator.mixed_precision == "fp16":
                weight_dtype = torch.float16
            elif accelerator.mixed_precision == "bf16":
                weight_dtype = torch.bfloat16
            else:
                weight_dtype = torch.float32

            with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
                # Generate images
                gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    predict_all_tokens=config.training.get("predict_all_tokens", False),
                    seq_len=config.model.showo.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )
            
            # Clamp the generated token ids to the correct range
            gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids).cpu()

            # Convert to PIL images
            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images += [Image.fromarray(image) for image in images]



        model.train()

        if config.training.get("pre_encode", False):
            del vq_model

        # # Convert to PIL images
        # images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        # images *= 255.0
        # images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        # pil_images = [Image.fromarray(image) for image in images]

        # Log images
        wandb_images = [wandb.Image(image, caption=validation_prompts[i]) for i, image in enumerate(pil_images)]
        wandb.log({"Generated images": wandb_images}, step=global_step)


def save_checkpoint(model, ema_model, config, accelerator, global_step):
    output_dir = config.experiment.output_dir
    checkpoints_total_limit = config.experiment.get("checkpoints_total_limit", None)

    # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
    if accelerator.is_main_process and checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]

            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)

    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    # XXX: could also make this conditional on deepspeed
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    state_dict = accelerator.get_state_dict(ema_model)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(ema_model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model_ema",
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()
