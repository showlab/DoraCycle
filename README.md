<p align="center">

<h2 align="center">DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles</h2>
<p align="center">
  <a href="https://ruizhaocv.github.io/"><strong>Rui Zhao</strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=S7bGBmkyNtEC&hl=en"><strong>Weijia Mao</strong></a>
  ·
  <a href="https://sites.google.com/view/showlab"><strong>Mike Zheng Shou</strong></a>
  <br>
  <br>
  <a href="https://arxiv.org/abs/2503.03651"><img src='https://img.shields.io/badge/arXiv-2503.03651-b31b1b.svg'></a>
  <br>
  <b>Show Lab, National University of Singapore</b>
</p>

<p align="center">
<img src="https://github.com/ruizhaocv/ruizhaocv.github.io/blob/master/images/DoraCycle_teaser.png" width="720px"/>  
<br>

---

This repository provides the official training and inference code for DoraCycle.

## Dependencies

- Base: [Show-o](https://github.com/showlab/Show-o) (512×512, MAGVITv2, Phi-1.5).
- Install: `pip install -r requirements.txt`; training uses `accelerate` and DeepSpeed.
- Optional: `wandb login` for logging.

## Pretrained Weights

- **MAGVITv2**: [showlab/magvitv2](https://huggingface.co/showlab/magvitv2)
- **Show-o 512×512**: [showlab/show-o-512x512](https://huggingface.co/showlab/show-o-512x512)
- **Phi-1.5**: [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5)

Set these (or local paths) in your config.

## Training

Single training entrypoint: **`training/train_doracycle.py`** with **`configs/doracycle.yaml`**.

1. Edit `configs/doracycle.yaml`: set `experiment.cus_data_path` to your unpaired image + text data path (see `training/cus_data_parque.py` / CusDataset for format). Set `experiment.characters_names` and, if needed, `dataset.params.train_lm_shards_path_or_url`.
2. Run:

```bash
accelerate launch --config_file accelerate_configs/8_gpu_deepspeed_zero2.yaml --main_process_port=8888 \
  training/train_doracycle.py config=configs/doracycle.yaml
```

Checkpoints are saved under `experiment.output_dir`. Use `resume_from_checkpoint: 'latest'` to resume.

## Inference

DoraCycle checkpoints are saved under `output_dir/checkpoint-XXXX/` with `unwrapped_model/` and `unwrapped_model_ema/`. Set `model.showo.pretrained_model_path` in config to the **checkpoint root** (e.g. `.../checkpoint-XXXX`).

**Text-to-image:** `inference_t2i_lora.py`

```bash
python3 inference_t2i_lora.py config=configs/doracycle.yaml \
  validation_prompts_file=validation_prompts/validation_prompts.txt
```

**Multimodal understanding:** `inference_mmu_lora.py`

```bash
python3 inference_mmu_lora.py config=configs/doracycle.yaml \
  mmu_image_root=./mmu_validation question='Please describe this image in details'
```

**Alternative — load full checkpoint**  
Use `inference_t2i.py` / `inference_mmu.py` and set `pretrained_model_path` to the model subfolder (e.g. `.../checkpoint-XXXX/unwrapped_model`).

## Citation

```bibtex
@article{zhao2025doracycle,
  title={DoraCycle: Domain-Oriented Adaptation of Unified Generative Model in Multimodal Cycles},
  author={Zhao, Rui and Mao, Weijia and Shou, Mike Zheng},
  journal={arXiv preprint arXiv:2503.03651},
  year={2025}
}

@article{xie2024showo,
  title={Show-o: One Single Transformer to Unify Multimodal Understanding and Generation},
  author={Xie, Jinheng and Mao, Weijia and Bai, Zechen and Zhang, David Junhao and others},
  journal={arXiv preprint arXiv:2408.12528},
  year={2024}
}
```
