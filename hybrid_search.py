import json
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


# =========================
# 1. 读取 chunks 和 embeddings
# =========================

with open(
    "data/chunks_v3.json",
    "r",
    encoding="utf-8"
) as f:
    chunks = json.load(f)

embeddings = np.load("data/embeddings.npy")

print("Chunks 数量：", len(chunks))
print("Embeddings shape：", embeddings.shape)
# =========================
# 2. 建立 BM25 索引
# =========================

def tokenize(text):
    """
    简单英文分词：
    转成小写，并提取英文单词和数字。
    """
    return re.findall(
        r"[a-zA-Z]+|\d+(?:\.\d+)?",
        text.lower()
    )


tokenized_corpus = [
    tokenize(chunk["text"])
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_corpus)

print("BM25 索引建立完成")
print("BM25 文档数量：", len(tokenized_corpus))
# =========================
# 3. Vector Search
# =========================

model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B"
)


def vector_search(query, top_k=20):
    """
    把问题转换成向量，
    再和所有 chunk embeddings 计算余弦相似度。
    """

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = embeddings @ query_embedding

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        results.append({
            "index": int(idx),
            "rank": rank,
            "score": float(scores[idx])
        })

    return results


print("Vector Search 准备完成")
# =========================
# 4. BM25 Search
# =========================

def bm25_search(query, top_k=20):
    """
    对用户问题进行分词，
    使用 BM25 对所有 chunks 进行关键词匹配排序。
    """

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        results.append({
            "index": int(idx),
            "rank": rank,
            "score": float(scores[idx])
        })

    return results


print("BM25 Search 准备完成")
# =========================
# 5. RRF Hybrid Search
# =========================

def hybrid_search(query, top_k=5, candidate_k=20, rrf_k=60):
    """
    使用 RRF（Reciprocal Rank Fusion）
    融合 Vector Search 和 BM25 Search 的排名。
    """

    vector_results = vector_search(
        query,
        top_k=candidate_k
    )

    bm25_results = bm25_search(
        query,
        top_k=candidate_k
    )

    rrf_scores = {}

    # Vector 排名贡献
    for result in vector_results:
        idx = result["index"]
        rank = result["rank"]

        rrf_scores[idx] = rrf_scores.get(idx, 0) + (
            1 / (rrf_k + rank)
        )

    # BM25 排名贡献
    for result in bm25_results:
        idx = result["index"]
        rank = result["rank"]

        rrf_scores[idx] = rrf_scores.get(idx, 0) + (
            1 / (rrf_k + rank)
        )

    ranked_indices = sorted(
        rrf_scores,
        key=rrf_scores.get,
        reverse=True
    )[:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):

        results.append({
            "index": idx,
            "rank": rank,
            "rrf_score": rrf_scores[idx],
            "chunk": chunks[idx]
        })

    return results
# =========================
# 6. 测试 Hybrid Search
# =========================

query = input("\n请输入你的问题：").strip()
results = hybrid_search(query)

print()
print("=" * 80)
print("Query:", query)
print("=" * 80)

for result in results:

    chunk = result["chunk"]

    print()
    print(
        f"Rank {result['rank']} | "
        f"RRF {result['rrf_score']:.6f}"
    )

    print("ID:", chunk["id"])
    print("Page:", chunk["page"])
    print("Section:", chunk["section"])

    print(
        "Text:",
        chunk["text"][:500].replace("\n", " ")
    )
