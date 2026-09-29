from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

print("正在加载模型...")

model = SentenceTransformer(
    MODEL_NAME,
    device="mps"
)

# 模拟财报里的三个 Chunk
documents = [
    "The combined ratio was 98.8%, representing a year-on-year increase of 1.0 pp.",

    "The total investment income was RMB34,937 million, representing a year-on-year increase of RMB14,130 million.",

    "The comprehensive solvency margin ratio was 232.6%.",

    """The Company adjusted the asset allocation flexibly.
    In 2024, taking the opportunity brought about by the recovery of capital market,
    the Company increased the investment in secondary equity assets at low prices,
    which led to a significant increase in the fair value changes through profit or loss
    of secondary equity assets. The total investment income was RMB34,937 million,
    representing a year-on-year increase of RMB14,130 million."""
]
# 我们的问题
question = "Why did investment income increase?"

print("正在生成 Embedding...")

document_embeddings = model.encode(documents)
question_embedding = model.encode(question)

# 计算问题与三个文档之间的余弦相似度
scores = cos_sim(question_embedding, document_embeddings)[0]

# 从高到低排序
results = sorted(
    zip(documents, scores),
    key=lambda x: float(x[1]),
    reverse=True
)

print("\n问题：", question)
print("\n===== 相似度排名 =====")

for rank, (document, score) in enumerate(results, start=1):
    print(f"\n第 {rank} 名")
    print(f"相似度：{float(score):.4f}")
    print(f"文本：{document}")
