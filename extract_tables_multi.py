import json
import re
from pathlib import Path

import pymupdf


REPORT_DIR = Path("documents/annual_reports")
OUTPUT_PATH = Path("data/tables_multi.json")


def parse_filename(path):
    match = re.match(r"(.+?)_(\d{4})_年报\.pdf$", path.name)

    if not match:
        raise ValueError(f"无法解析文件名：{path.name}")

    company = match.group(1)
    year = int(match.group(2))

    return company, year


def clean_cell(cell):
    """
    清洗表格单元格：
    1. None 保留为空字符串
    2. 单元格内部换行替换为空格
    3. 合并连续空格
    """
    if cell is None:
        return ""

    text = str(cell)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def table_to_text(rows):
    """
    把二维表格转换成适合 RAG 检索的文本。
    保留行和列之间的对应关系。
    """
    lines = []

    for row in rows:
        cleaned_row = [clean_cell(cell) for cell in row]

        if any(cleaned_row):
            lines.append(" | ".join(cleaned_row))

    return "\n".join(lines)


def extract_report_tables(path):
    company, year = parse_filename(path)

    print("\n" + "=" * 70)
    print(f"正在处理：{company} {year}")
    print("=" * 70)

    doc = pymupdf.open(path)

    report_tables = []

    for page_index, page in enumerate(doc):

        page_number = page_index + 1

        try:
            result = page.find_tables()
        except Exception as e:
            print(f"PDF第 {page_number} 页表格识别失败：{e}")
            continue

        for table_index, table in enumerate(result.tables, start=1):

            try:
                raw_rows = table.extract()
            except Exception:
                continue

            if not raw_rows:
                continue

            rows = []

            for row in raw_rows:
                cleaned_row = [
                    clean_cell(cell)
                    for cell in row
                ]

                rows.append(cleaned_row)

            table_text = table_to_text(rows)

            if not table_text.strip():
                continue

            max_columns = max(
                (len(row) for row in rows),
                default=0
            )

            table_data = {
                "id": (
                    f"{company}_{year}_"
                    f"p{page_number}_table{table_index}"
                ),
                "company": company,
                "year": year,
                "page": page_number,
                "source": path.name,
                "table_index": table_index,
                "row_count": len(rows),
                "column_count": max_columns,
                "rows": rows,
                "text": table_text,
            }

            report_tables.append(table_data)

    doc.close()

    print(f"提取表格：{len(report_tables)} 张")

    return report_tables


def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_paths = sorted(
        REPORT_DIR.glob("*.pdf")
    )

    print(f"共发现 {len(pdf_paths)} 份报告")

    all_tables = []

    for path in pdf_paths:
        tables = extract_report_tables(path)
        all_tables.extend(tables)

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_tables,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 70)
    print("全部表格提取完成")
    print("=" * 70)

    print(f"报告数量：{len(pdf_paths)}")
    print(f"表格总数：{len(all_tables)}")
    print(f"已保存：{OUTPUT_PATH}")

    print("\n各公司表格数量：")

    company_counts = {}

    for table in all_tables:
        company = table["company"]

        company_counts[company] = (
            company_counts.get(company, 0) + 1
        )

    for company in sorted(company_counts):
        print(
            f"  {company}: "
            f"{company_counts[company]}"
        )


if __name__ == "__main__":
    main()
