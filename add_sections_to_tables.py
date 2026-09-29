import json
from collections import defaultdict, Counter


CHUNKS_PATH = "data/chunks_multi.json"
TABLES_PATH = "data/tables_multi.json"
OUTPUT_PATH = "data/tables_multi_with_sections.json"


def main():

    # 读取正文 Chunk
    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    # 读取表格
    with open(
        TABLES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        tables = json.load(f)

    # 建立：
    # (公司, 页码) -> [该页出现的章节]
    page_sections = defaultdict(list)

    for chunk in chunks:

        key = (
            chunk["company"],
            chunk["page"]
        )

        section = chunk["section"]

        if (
            section
            and section != "Unknown"
            and section not in page_sections[key]
        ):
            page_sections[key].append(section)

    unknown_count = 0
    multi_section_count = 0

    for table in tables:

        key = (
            table["company"],
            table["page"]
        )

        sections = page_sections.get(key, [])

        if len(sections) == 0:

            table["section"] = "Unknown"
            unknown_count += 1

        elif len(sections) == 1:

            table["section"] = sections[0]

        else:
            # 同一页存在多个一级章节时，
            # 暂时记录全部候选章节，不武断指定其中一个。
            table["section"] = " / ".join(sections)
            multi_section_count += 1

        # 为后续 RAG 准备结构化文本
        table["embedding_text"] = (
            f"公司：{table['company']}\n"
            f"年份：{table['year']}\n"
            f"章节：{table['section']}\n"
            f"PDF页码：{table['page']}\n"
            f"类型：表格\n"
            f"{table['text']}"
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            tables,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("=" * 70)
    print("表格章节标注完成")
    print("=" * 70)

    print(f"表格总数：{len(tables)}")
    print(f"Unknown：{unknown_count}")
    print(f"同页多章节：{multi_section_count}")
    print(f"已保存：{OUTPUT_PATH}")

    print("\n各公司 Section 情况：")

    company_stats = defaultdict(Counter)

    for table in tables:
        company_stats[table["company"]][table["section"]] += 1

    for company in sorted(company_stats):

        print("\n" + company)

        for section, count in company_stats[company].most_common():

            print(
                f"  {count:>4}  {section}"
            )


if __name__ == "__main__":
    main()
