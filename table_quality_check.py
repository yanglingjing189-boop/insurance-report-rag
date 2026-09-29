import json
from collections import Counter, defaultdict


INPUT_PATH = "data/tables_multi_with_sections.json"


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

    # 有多少行只有一个非空单元格
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

    # 规则1：只有1列
    if column_count <= 1:
        reasons.append("只有1列")

    # 规则2：表格太小
    if row_count <= 1:
        reasons.append("只有1行")

    # 规则3：空单元格过多
    if empty_ratio >= 0.60:
        reasons.append(
            f"空单元格比例高({empty_ratio:.0%})"
        )

    # 规则4：多数行只有一个有效单元格，
    # 可能出现整行内容被挤进一列
    if (
        row_count >= 3
        and single_cell_ratio >= 0.50
    ):
        reasons.append(
            f"单有效单元格行比例高({single_cell_ratio:.0%})"
        )

    status = (
        "suspicious"
        if reasons
        else "pass"
    )

    return {
        "status": status,
        "reasons": reasons,
        "empty_ratio": empty_ratio,
        "single_cell_ratio": single_cell_ratio,
    }


def main():

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        tables = json.load(f)

    status_counts = Counter()
    company_stats = defaultdict(Counter)
    suspicious = []

    for table in tables:

        result = analyze_table(table)

        status = result["status"]

        status_counts[status] += 1
        company_stats[table["company"]][status] += 1

        if status == "suspicious":
            suspicious.append({
                "id": table["id"],
                "company": table["company"],
                "page": table["page"],
                "section": table["section"],
                "row_count": table["row_count"],
                "column_count": table["column_count"],
                "reasons": result["reasons"],
                "text_preview": table["text"][:300],
            })

    print("=" * 70)
    print("表格质量体检")
    print("=" * 70)

    print(f"表格总数：{len(tables)}")
    print(f"初步通过：{status_counts['pass']}")
    print(f"可疑表格：{status_counts['suspicious']}")

    print("\n各公司情况：")

    for company in sorted(company_stats):

        passed = company_stats[company]["pass"]
        suspicious_count = (
            company_stats[company]["suspicious"]
        )

        print(
            f"{company}: "
            f"通过 {passed}，"
            f"可疑 {suspicious_count}"
        )

    print("\n可疑原因统计：")

    reason_counts = Counter()

    for item in suspicious:
        for reason in item["reasons"]:
            reason_type = reason.split("(")[0]
            reason_counts[reason_type] += 1

    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")

    print("\n前 20 张可疑表格：")

    for item in suspicious[:20]:

        print("\n" + "-" * 70)
        print("ID：", item["id"])
        print("公司：", item["company"])
        print("PDF页码：", item["page"])
        print("章节：", item["section"])
        print(
            "行 × 列：",
            item["row_count"],
            "×",
            item["column_count"]
        )
        print(
            "原因：",
            "；".join(item["reasons"])
        )
        print("内容预览：")
        print(item["text_preview"])


if __name__ == "__main__":
    main()
