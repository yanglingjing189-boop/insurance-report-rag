import json
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/chunks_v3.json"

EMBEDDINGS_WITH_METADATA = "data/embeddings.npy"
EMBEDDINGS_TEXT_ONLY = "data/embeddings_text_only.npy"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
TOP_K = 5


# 读取 Chunk
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)


# 读取两套 Embedding
embeddings_a = np.load(EMBEDDINGS_WITH_METADATA)
embeddings_b = np.load(EMBEDDINGS_TEXT_ONLY)

print("Chunk 数量：", len(chunks))
print("A shape：", embeddings_a.shape)
print("B shape：", embeddings_b.shape)


# 加载模型
print("\n正在加载 Qwen Embedding 模型...")

model = SentenceTransformer(
    MODEL_NAME,
    device="mps"
)


# 输入问题
question = input("\n请输入问题：").strip()

question_embedding = model.encode(
    question,
    normalize_embeddings=True
)
def search(embeddings):
    scores = embeddings @ question_embedding

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = []

    for index in top_indices:
        results.append({
            "index": index,
            "score": scores[index],
            "chunk": chunks[index]
        })

    return results


results_a = search(embeddings_a)
results_b = search(embeddings_b)


def print_results(title, results):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        chunk = result["chunk"]

        print(f"\n第 {rank} 名")
        print(f"相似度：{result['score']:.4f}")
        print(f"Chunk ID：{chunk['id']}")
        print(f"PDF页码：{chunk['page']}")

        # 为了方便比较，只显示前300个字符
        preview = chunk["text"][:300].replace("\n", " ")

        print("内容预览：")
        print(preview)


print_results(
    "实验 A：Metadata + 正文",
    results_a
)

print_results(
    "实验 B：纯正文",
    results_b
)
