import pymupdf

PDF_PATH = "documents/人保财险2024年报.pdf"

doc = pymupdf.open(PDF_PATH)

for page_number in [20, 22]:

    page = doc[page_number - 1]
    tables = page.find_tables()

    print(f"\nPDF 第 {page_number} 页")
    print("检测到表格数量：", len(tables.tables))

    for i, table in enumerate(tables.tables, start=1):
        print(f"表格 {i} bbox：", table.bbox)
