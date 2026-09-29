import pymupdf

pdf_path = "documents/人保财险2024年报.pdf"

doc = pymupdf.open(pdf_path)

page_number = 20

page = doc[page_number - 1]
text = page.get_text()

print(f"========== PDF第 {page_number} 页 ==========\n")
print(text)
