import numpy as np

# paths = ["imagenet_benchmarks/imagenet256_0_tp_10_device_0_step_340000_generate3_only_generation.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_10_device_1_step_340000_generate3_only_generation.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_10_device_2_step_340000_generate3_only_generation.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_10_device_3_step_340000_generate3_only_generation.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_10_device_4_step_340000_generate3_only_generation.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_10_device_5_step_340000_generate3_only_generation.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_10_device_6_step_340000_generate3_only_generation.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_10_device_7_step_340000_generate3_only_generation.npz",]
#
# images = []
# for path in paths:
#     with open(path, "rb") as f:
#         data = np.load(f)
#         images.append(data['arr_0'])
# images = np.concatenate(images, axis=0)
# print(images.shape)
# np.savez("imagenet_benchmarks/imagenet256_0_tp_10_step_340000_generate3_only_generation.npz", images)

# paths = ["imagenet_benchmarks/imagenet256_0_tp_4_device_0_step_400000_generate3_only_generation_llamagen_seq.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_4_device_1_step_400000_generate3_only_generation_llamagen_seq.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_4_device_2_step_400000_generate3_only_generation_llamagen_seq.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_4_device_3_step_400000_generate3_only_generation_llamagen_seq.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_4_device_4_step_400000_generate3_only_generation_llamagen_seq.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_4_device_5_step_400000_generate3_only_generation_llamagen_seq.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_4_device_6_step_400000_generate3_only_generation_llamagen_seq.npz",
#         "imagenet_benchmarks/imagenet256_0_tp_4_device_7_step_400000_generate3_only_generation_llamagen_seq.npz",]
#
# images = []
# for path in paths:
#     with open(path, "rb") as f:
#         data = np.load(f)
#         images.append(data['arr_0'])
# images = np.concatenate(images, axis=0)
# print(images.shape)
# np.savez("imagenet_benchmarks/imagenet256_0_tp_4_step_400000_generate3_only_generation_llamagen_seq.npz", images)
#


# paths = ["imagenet_benchmarks/imagenet256_1.75_tp_10_device_0_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz",
#         "imagenet_benchmarks/imagenet256_1.75_tp_10_device_1_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz",
#         "imagenet_benchmarks/imagenet256_1.75_tp_10_device_2_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz",
#         "imagenet_benchmarks/imagenet256_1.75_tp_10_device_3_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz",
#         "imagenet_benchmarks/imagenet256_1.75_tp_10_device_4_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz",
#         "imagenet_benchmarks/imagenet256_1.75_tp_10_device_5_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz",
#         "imagenet_benchmarks/imagenet256_1.75_tp_10_device_6_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz",
#         "imagenet_benchmarks/imagenet256_1.75_tp_10_device_7_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz",]
#
# images = []
# for path in paths:
#     with open(path, "rb") as f:
#         data = np.load(f)
#         images.append(data['arr_0'])
# images = np.concatenate(images, axis=0)
# print(images.shape)
# np.savez("imagenet_benchmarks/imagenet256_1.75_tp_10_step_90000_generate3_only_generation_llamagen_seq_cfg_token_five.npz", images)

#imagenet_benchmarks/mscoco256_1.8_tp_8_step_61000_generate3
paths = ["./coco30k_benchmarks/mscoco512_1.75_tp_18_device_0_step_checkpoint-70000.npz",
        "./coco30k_benchmarks/mscoco512_1.75_tp_18_device_1_step_checkpoint-70000.npz",
        "./coco30k_benchmarks/mscoco512_1.75_tp_18_device_2_step_checkpoint-70000.npz",
        "./coco30k_benchmarks/mscoco512_1.75_tp_18_device_3_step_checkpoint-70000.npz",
        "./coco30k_benchmarks/mscoco512_1.75_tp_18_device_4_step_checkpoint-70000.npz",
        "./coco30k_benchmarks/mscoco512_1.75_tp_18_device_5_step_checkpoint-70000.npz",
        "./coco30k_benchmarks/mscoco512_1.75_tp_18_device_6_step_checkpoint-70000.npz",
        "./coco30k_benchmarks/mscoco512_1.75_tp_18_device_7_step_checkpoint-70000.npz",]

images = []
for path in paths:
    with open(path, "rb") as f:
        data = np.load(f)
        images.append(data['arr_0'])
images = np.concatenate(images, axis=0)
print(images.shape)
np.savez("./coco30k_benchmarks/mscoco512_1.75_tp_18_step_checkpoint-70000.npz", images)
