import json

INPUT_PATH = "data/tables.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    pages = json.load(f)

# 找到 PDF 第20页
page = next(p for p in pages if p["page"] == 22)
words = page["words"]

# 这张表格的数据区域
TABLE_TOP = 210
TABLE_BOTTOM = 360

# 根据第20页和第22页观察到的版式划分四列
COL_1_END = 280
COL_2_END = 355
COL_3_END = 425


def get_column(x):
    if x < COL_1_END:
        return 0
    elif x < COL_2_END:
        return 1
    elif x < COL_3_END:
        return 2
    else:
        return 3


# 只保留表格区域里的词
table_words = [
    w for w in words
    if TABLE_TOP <= w["y0"] <= TABLE_BOTTOM
]

# ---------- 按 y 坐标恢复行 ----------

rows = []

for word in sorted(table_words, key=lambda w: (w["y0"], w["x0"])):

    # PDF中的同一行文字，y坐标可能相差零点几
    matching_row = None

    for row in rows:
        if abs(row["y"] - word["y0"]) <= 1.0:
            matching_row = row
            break

    if matching_row is None:
        matching_row = {
            "y": word["y0"],
            "cells": [[], [], [], []]
        }
        rows.append(matching_row)

    column = get_column(word["x0"])
    matching_row["cells"][column].append(word)


# ---------- 每个单元格内部按 x 排序并拼接 ----------

result = []

for row in rows:

    cells = []

    for cell_words in row["cells"]:
        cell_words.sort(key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in cell_words)
        cells.append(text)

    result.append(cells)

# ---------- 合并跨视觉行的指标名称 ----------

merged_result = []

i = 0

while i < len(result):

    row = result[i]

    # 如果这一行只有第一列有内容，
    # 而三个数据列全部为空，
    # 则尝试与下一行合并
    if (
        row[0]
        and not row[1]
        and not row[2]
        and not row[3]
        and i + 1 < len(result)
    ):
        next_row = result[i + 1]

        # 下一行有指标名称，并且至少有一个数据列
        if (
            next_row[0]
            and (next_row[1] or next_row[2] or next_row[3])
        ):
            merged_row = [
                row[0] + " " + next_row[0],
                next_row[1],
                next_row[2],
                next_row[3]
            ]

            merged_result.append(merged_row)

            i += 2
            continue

    merged_result.append(row)
    i += 1

result = merged_result
# ---------- 输出结果 ----------

print("PDF页码：", page["page"])
print("章节：", page["section"])
print()

print(
    f"{'Item':<35} | "
    f"{'2024':<12} | "
    f"{'2023':<12} | "
    f"Change"
)

print("-" * 85)

for row in result:
    print(
        f"{row[0]:<35} | "
        f"{row[1]:<12} | "
        f"{row[2]:<12} | "
        f"{row[3]}"
    )
