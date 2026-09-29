import pymupdf

PDF_PATH = "documents/人保财险2024年报.pdf"

doc = pymupdf.open(PDF_PATH)

for page_number in [20, 22]:

    page = doc[page_number - 1]
    tables = page.find_tables()

    print(f"\n===== PDF 第 {page_number} 页 =====")

    for table_index, table in enumerate(tables.tables, start=1):

        x0, y0, x1, y1 = table.bbox

        print(f"表格 {table_index}")
        print("bbox:", table.bbox)

        # 获取整页文字坐标
        words = page.get_text("words")

        # 只保留表格 bbox 内的文字
        table_words = [
            w for w in words
            if x0 <= w[0] <= x1 
            and y0 - 40 <= w[1] <= y1
        ]

        # 查看表格顶部约 35 像素范围
        header_words = [
            w for w in table_words
            if w[1] < y0
        ]

        for word in sorted(header_words, key=lambda w: (w[1], w[0])):
            print(
                f"x={word[0]:7.1f} "
                f"y={word[1]:7.1f} "
                f"text={word[4]}"
            )
