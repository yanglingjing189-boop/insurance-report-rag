import pymupdf

PDF_PATH = "documents/人保财险2024年报.pdf"

doc = pymupdf.open(PDF_PATH)

# PDF第20页
page = doc[19]

words = page.get_text("words")

print("第22页词语数量：", len(words))
print()

for word in words:
    x0, y0, x1, y1, text, block_no, line_no, word_no = word

    # 只显示页面中部区域，减少无关内容
    if 180 < y0 < 380:
        print(
            f"x={x0:7.1f} "
            f"y={y0:7.1f} "
            f"text={text}"
        )
