from T2IBenchmark import calculate_coco_fid

# fid, fid_data = calculate_coco_fid(
#     PixelartWrapper,
#     device='cuda:2',
#     save_generations_dir='coco_generations/'
# )

import os
from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats

fid, _ = calculate_fid(
    os.environ.get("COCO_GENERATIONS_DIR", "<YOUR_COCO_GENERATIONS_DIR>"),
    os.environ.get("COCO_GROUND_TRUTH_DIR", "<YOUR_COCO_GROUND_TRUTH_DIR>"),
)
print(fid)