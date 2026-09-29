import pymupdf
import pymupdf.layout

PDF_PATH = "documents/人保财险2024年报.pdf"

doc = pymupdf.open(PDF_PATH)

# PDF第22页，对应索引21
page = doc[21]

tables = page.find_tables()

print("检测到表格数量：", len(tables.tables))

for i, table in enumerate(tables.tables, start=1):

    print("\n" + "=" * 70)
    print(f"表格 {i}")
    print("=" * 70)

    data = table.extract()

    for row in data:
        print(row)
