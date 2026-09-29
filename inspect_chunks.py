import json

FILE = "data/chunks_v3.json"

with open(FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

keyword = input("请输入要搜索的关键词：").strip()

results = [
    chunk for chunk in chunks
    if keyword.lower() in chunk["text"].lower()
]

print(f"\n共找到 {len(results)} 个 Chunk")

for chunk in results[:5]:
    print("\n" + "=" * 70)
    print("ID：", chunk["id"])
    print("PDF页码：", chunk["page"])
    print("字符数：", chunk["char_count"])
    print("-" * 70)
    print(chunk["text"])
