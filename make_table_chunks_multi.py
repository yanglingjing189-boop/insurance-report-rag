import json


INPUT_PATH = "data/tables_multi_with_sections.json"
OUTPUT_PATH = "data/table_chunks_multi.json"

MAX_LEN = 1400


def analyze_table(table):
    rows = table["rows"]

    total_cells = 0
    empty_cells = 0
    nonempty_per_row = []

    for row in rows:
        nonempty = 0

        for cell in row:
            total_cells += 1

            if cell is None or str(cell).strip() == "":
                empty_cells += 1
            else:
                nonempty += 1

        nonempty_per_row.append(nonempty)

    empty_ratio = (
        empty_cells / total_cells
        if total_cells > 0
        else 1
    )

    row_count = table["row_count"]
    column_count = table["column_count"]

    single_cell_rows = sum(
        n == 1
        for n in nonempty_per_row
    )

    single_cell_ratio = (
        single_cell_rows / row_count
        if row_count > 0
        else 1
    )

    reasons = []

    if column_count <= 1:
        reasons.append("只有1列")

    if row_count <= 1:
        reasons.append("只有1行")

    if empty_ratio >= 0.60:
        reasons.append("空单元格比例高")

    if (
        row_count >= 3
        and single_cell_ratio >= 0.50
    ):
        reasons.append("单有效单元格行比例高")

    return reasons


def row_to_text(row):
    return " | ".join(
        "" if cell is None else str(cell).strip()
        for cell in row
    )


def make_embedding_text(table, text):
    return (
        f"公司：{table['company']}\n"
        f"年份：{table['year']}\n"
        f"章节：{table['section']}\n"
        f"PDF页码：{table['page']}\n"
        f"类型：表格\n"
        f"{text}"
    )


def split_table(table):
    """
    短表：整张保留。

    长表：
    默认把第一行作为表头，
    每个子 Chunk 都重复表头，
    然后按数据行累积到 MAX_LEN 附近。

    注意：
    对复杂多层表头，这只是保守的通用切法；
    原始 rows 仍保存在 tables_multi_with_sections.json 中。
    """

    full_text = table["text"]

    if len(full_text) <= MAX_LEN:
        return [{
            "text": full_text,
            "rows": table["rows"],
        }]

    rows = table["rows"]

    if len(rows) <= 1:
        return [{
            "text": full_text,
            "rows": rows,
        }]

    header = rows[0]
    header_text = row_to_text(header)

    parts = []
    current_rows = []
    current_text = header_text

    for row in rows[1:]:

        row_text = row_to_text(row)

        candidate = (
            current_text
            + "\n"
            + row_text
        )

        if (
            len(candidate) > MAX_LEN
            and current_rows
        ):
            part_rows = [header] + current_rows

            parts.append({
                "text": current_text,
                "rows": part_rows,
            })

            current_rows = [row]
            current_text = (
                header_text
                + "\n"
                + row_text
            )

        else:
            current_rows.append(row)
            current_text = candidate

    if current_rows:
        part_rows = [header] + current_rows

        parts.append({
            "text": current_text,
            "rows": part_rows,
        })

    return parts


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        tables = json.load(f)

    table_chunks = []
    skipped = []

    accepted_tables = 0
    split_tables = 0

    for table in tables:

        reasons = analyze_table(table)

        if reasons:
            skipped.append({
                "id": table["id"],
                "company": table["company"],
                "page": table["page"],
                "reasons": reasons,
            })
            continue

        accepted_tables += 1

        parts = split_table(table)

        if len(parts) > 1:
            split_tables += 1

        for part_index, part in enumerate(
            parts,
            start=1
        ):

            if len(parts) == 1:
                chunk_id = table["id"]
            else:
                chunk_id = (
                    f"{table['id']}"
                    f"_part{part_index}"
                )

            chunk = {
                "id": chunk_id,
                "table_id": table["id"],
                "company": table["company"],
                "year": table["year"],
                "section": table["section"],
                "page": table["page"],
                "source": table["source"],
                "type": "table",
                "part": part_index,
                "part_count": len(parts),
                "text": part["text"],
                "embedding_text": make_embedding_text(
                    table,
                    part["text"]
                ),
                "row_count": len(part["rows"]),
                "column_count": table["column_count"],
                "rows": part["rows"],
            }

            table_chunks.append(chunk)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            table_chunks,
            f,
            ensure_ascii=False,
            indent=2
        )

    lengths = [
        len(chunk["text"])
        for chunk in table_chunks
    ]

    print("=" * 70)
    print("Table Chunk 生成完成")
    print("=" * 70)

    print(f"原始候选表格：{len(tables)}")
    print(f"质量检查通过表格：{accepted_tables}")
    print(f"其中需要切分的长表：{split_tables}")
    print(f"暂不进入 RAG：{len(skipped)}")
    print(f"最终 Table Chunk：{len(table_chunks)}")
    print(f"已保存：{OUTPUT_PATH}")

    if lengths:
        print("\n最终 Chunk 长度：")
        print(f"最短：{min(lengths)}")
        print(f"最长：{max(lengths)}")
        print(
            "超过 MAX_LEN：",
            sum(
                length > MAX_LEN
                for length in lengths
            )
        )


if __name__ == "__main__":
    main()
