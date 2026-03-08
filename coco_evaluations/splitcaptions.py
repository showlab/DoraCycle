from T2IBenchmark.datasets import get_coco_30k_captions, get_coco_fid_stats
import math
id2caption = get_coco_30k_captions()
captions = []
ids = []
for d in id2caption.items():
    ids.append(d[0])
    captions.append(d[1])

def split_data_into_chunks(data, num_chunks):
    """
    将 data 列表分成指定数量的 chunk
    :param data: List of data
    :param num_chunks: Number of chunks
    :return: List of data chunks
    """
    total_data = len(data)
    chunk_size = math.ceil(total_data / num_chunks)
    return [data[i:i + chunk_size] for i in range(0, total_data, chunk_size)]



# 分成8个chunks
num_chunks = 8
id_chunks = split_data_into_chunks(ids, num_chunks)
caption_chunks = split_data_into_chunks(captions, num_chunks)

# 确保 id 和 caption 是对应的
id_caption_chunks = [(id_chunks[i], caption_chunks[i]) for i in range(num_chunks)]

# 打印结果
for idx, (id_chunk, caption_chunk) in enumerate(id_caption_chunks):
    print(f"Chunk {idx+1}:")
    for id_, caption in zip(id_chunk, caption_chunk):
        print(f"ID: {id_}, Caption: {caption}")