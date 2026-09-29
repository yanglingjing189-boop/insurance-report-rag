import json
from collections import Counter


TEXT_PATH = "data/chunks_multi.json"
TABLE_PATH = "data/table_chunks_multi.json"
OUTPUT_PATH = "data/chunks_rag_multi.json"


def main():

    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text_chunks = json.load(f)

    with open(TABLE_PATH, "r", encoding="utf-8") as f:
        table_chunks = json.load(f)

    # 给正文 Chunk 明确增加 type 字段
    for chunk in text_chunks:
        chunk["type"] = "text"

    # 表格 Chunk 已经有 type="table"
    all_chunks = text_chunks + table_chunks

    # 最终检查
    ids = [chunk["id"] for chunk in all_chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("发现重复 Chunk ID")

    for chunk in all_chunks:
        if not str(chunk.get("embedding_text", "")).strip():
            raise ValueError(
                f"embedding_text 为空：{chunk['id']}"
            )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            all_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    type_counts = Counter(
        chunk["type"]
        for chunk in all_chunks
    )

    company_counts = Counter(
        chunk["company"]
        for chunk in all_chunks
    )

    print("=" * 70)
    print("多公司 RAG Chunk 合并完成")
    print("=" * 70)

    print(f"正文 Chunk：{type_counts['text']}")
    print(f"表格 Chunk：{type_counts['table']}")
    print(f"总 Chunk：{len(all_chunks)}")
    print(f"已保存：{OUTPUT_PATH}")

    print("\n各公司：")
    for company, count in sorted(company_counts.items()):
        print(f"  {company}: {count}")


if __name__ == "__main__":
    main()
