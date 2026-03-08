import os
from T2IBenchmark import calculate_coco_fid

# fid, fid_data = calculate_coco_fid(
#     PixelartWrapper,
#     device='cuda:2',
#     save_generations_dir='coco_generations/'
# )

from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats
# a = get_coco_fid_stats()
# import ipdb
# ipdb.set_trace()
fid, _ = calculate_fid(
    os.environ.get("COCO_GENERATIONS_DIR", "<YOUR_COCO_GENERATIONS_DIR>"),
    get_coco_fid_stats(),
)
# fid, _ = calculate_fid(
#     'imagenet_benchmarks/mscoco256_1.8_tp_8_step_61000_generate3.npz',
#     get_coco_fid_stats()
# )