import json
import numpy as np
import torch

from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/chunks_rag_multi.json"
OUTPUT_PATH = "data/embeddings_multi.npy"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
BATCH_SIZE = 16


def main():

    print("=" * 70)
    print("多公司 Embedding 生成")
    print("=" * 70)

    # 1. 读取 Chunk
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    texts = [
        chunk["embedding_text"]
        for chunk in chunks
    ]

    print(f"Chunk 数量：{len(chunks)}")

    # 2. 自动选择设备
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"计算设备：{device}")
    print(f"Embedding 模型：{MODEL_NAME}")

    # 3. 加载模型
    print("\n正在加载 Embedding 模型...")

    model = SentenceTransformer(
        MODEL_NAME,
        device=device
    )

    # 4. 生成 Embedding
    print("正在生成向量...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 5. 基础检查
    if len(embeddings) != len(chunks):
        raise ValueError(
            "Embedding 数量与 Chunk 数量不一致"
        )

    # 6. 保存
    np.save(
        OUTPUT_PATH,
        embeddings
    )

    print("\n" + "=" * 70)
    print("Embedding 生成完成")
    print("=" * 70)

    print(f"Chunk 数量：{len(chunks)}")
    print(f"Embedding shape：{embeddings.shape}")
    print(f"数据类型：{embeddings.dtype}")
    print(f"已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
