from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

print("正在加载模型...")

model = SentenceTransformer(
    MODEL_NAME,
    device="mps"
)

texts = [
    "The combined ratio was 98.8%.",
    "The total investment income was RMB34,937 million."
]

print("正在生成 Embedding...")

embeddings = model.encode(texts)

print("\nEmbedding shape:", embeddings.shape)

print("\n第一句话：")
print(texts[0])

print("\n前10个向量数字：")
print(embeddings[0][:10])

print("\n第二句话：")
print(texts[1])

print("\n前10个向量数字：")
print(embeddings[1][:10])
