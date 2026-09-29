import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks_v3.json"
OUTPUT_PATH = "data/embeddings_text_only.npy"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

print("正在读取 Chunks...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# 这次只使用财报正文，不加入 company / year / page
texts = [chunk["text"] for chunk in chunks]

print("Chunk 数量：", len(chunks))

print("\n正在加载 Qwen Embedding 模型...")

model = SentenceTransformer(
    MODEL_NAME,
    device="mps"
)

print("\n正在生成纯正文 Embeddings...")

embeddings = model.encode(
    texts,
    batch_size=8,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEmbedding shape：", embeddings.shape)

np.save(OUTPUT_PATH, embeddings)

print("保存位置：", OUTPUT_PATH)
print("纯正文 Embedding 构建完成！")
