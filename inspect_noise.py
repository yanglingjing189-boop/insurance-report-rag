import json

CHUNKS_PATH = "data/chunks_v3.json"

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

short_chunks = [
    chunk for chunk in chunks
    if len(chunk["text"].strip()) < 200
]

print("全部 Chunk：", len(chunks))
print("少于200字符：", len(short_chunks))

print("\n===== 短 Chunk 样本 =====")

for chunk in short_chunks[:20]:
    print("\n" + "=" * 70)
    print("ID：", chunk["id"])
    print("PDF页码：", chunk["page"])
    print("字符数：", len(chunk["text"].strip()))
    print("-" * 70)
    print(chunk["text"])

