import json
import re
import numpy as np
from rank_bm25 import BM25Okapi


# =========================
# 1. 读取 Chunks
# =========================

CHUNKS_PATH = "data/chunks_v3.json"

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("Chunk 数量：", len(chunks))


# =========================
# 2. 简单英文分词
# =========================

def tokenize(text):
    text = text.lower()

    # 提取英文单词和数字
    tokens = re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", text)

    return tokens


# =========================
# 3. 构建 BM25 索引
# =========================

print("正在构建 BM25 索引...")

corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(corpus)

print("BM25 索引构建完成！")


# =========================
# 4. 输入问题
# =========================

question = input("\n请输入问题：\n")

query_tokens = tokenize(question)

scores = bm25.get_scores(query_tokens)


# =========================
# 5. 获取 Top 5
# =========================

top_k = 5

top_indices = np.argsort(scores)[::-1][:top_k]

print("\n" + "=" * 70)
print("BM25 Top 5")
print("=" * 70)


for rank, idx in enumerate(top_indices, start=1):

    chunk = chunks[idx]

    print(f"\n第 {rank} 名")
    print(f"BM25分数：{scores[idx]:.4f}")
    print(f"公司：{chunk['company']}")
    print(f"年份：{chunk['year']}")
    print(f"PDF页码：{chunk['page']}")
    print(f"Chunk ID：{chunk['id']}")

    print("-" * 70)

    print(chunk["text"])
