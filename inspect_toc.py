import fitz

PDF_PATH = "documents/人保财险2024年报.pdf"

doc = fitz.open(PDF_PATH)

toc = doc.get_toc()

print("目录条目数量：", len(toc))
print()

for item in toc:
    level, title, page = item

    print(
        f"层级 {level} | "
        f"PDF页码 {page} | "
        f"{title}"
    )
