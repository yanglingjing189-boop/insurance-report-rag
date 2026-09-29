import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks_v3.json"
EMBEDDINGS_PATH = "data/embeddings.npy"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

TOP_K = 5

print("正在加载知识库...")

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(EMBEDDINGS_PATH)

print("Chunk 数量：", len(chunks))
print("Embedding shape：", embeddings.shape)

print("\n正在加载 Qwen Embedding 模型...")

model = SentenceTransformer(
    MODEL_NAME,
    device="mps"
)

question = input("\n请输入问题：").strip()

print("\n正在检索...")

question_embedding = model.encode(
    question,
    normalize_embeddings=True
)

# 因为所有向量都已经归一化，
# 点积就等价于余弦相似度
scores = embeddings @ question_embedding

# 按相似度从高到低，取前 TOP_K 个
top_indices = np.argsort(scores)[::-1][:TOP_K]

print("\n" + "=" * 70)
print("问题：", question)
print("=" * 70)

for rank, index in enumerate(top_indices, start=1):
    chunk = chunks[index]
    score = scores[index]

    print(f"\n第 {rank} 名")
    print(f"相似度：{score:.4f}")
    print(f"公司：{chunk['company']}")
    print(f"年份：{chunk['year']}")
    print(f"PDF页码：{chunk['page']}")
    print(f"Chunk ID：{chunk['id']}")
    print("-" * 70)
    print(chunk["text"])
