import pymupdf
import json
import os

PDF_PATH = "documents/人保财险2024年报.pdf"
OUTPUT_PATH = "data/tables.json"

COMPANY = "人保财险"
YEAR = 2024
SOURCE = "人保财险2024年报.pdf"

doc = pymupdf.open(PDF_PATH)

# ---------- 读取官方目录，用于确定章节 ----------

toc = doc.get_toc()

sections = []

for level, title, page in toc:
    if level == 1:
        sections.append({
            "title": title,
            "page": page
        })


def get_section(page_number):
    current_section = "Unknown"

    for section in sections:
        if page_number >= section["page"]:
            current_section = section["title"]
        else:
            break

    return current_section


# ---------- 第一阶段：先提取每页所有词及坐标 ----------

pages = []

for page_index, page in enumerate(doc):

    page_number = page_index + 1
    section = get_section(page_number)

    words = page.get_text("words")

    word_items = []

    for word in words:
        x0, y0, x1, y1, text, block_no, line_no, word_no = word

        word_items.append({
            "text": text,
            "x0": round(x0, 1),
            "y0": round(y0, 1),
            "x1": round(x1, 1),
            "y1": round(y1, 1)
        })

    pages.append({
        "company": COMPANY,
        "year": YEAR,
        "page": page_number,
        "section": section,
        "source": SOURCE,
        "words": word_items
    })


os.makedirs("data", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(pages, f, ensure_ascii=False, indent=2)


print("坐标数据提取完成！")
print("PDF页数：", len(doc))
print("保存位置：", OUTPUT_PATH)
