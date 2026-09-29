import fitz

PDF_PATH = "documents/人保财险2024年报.pdf"

doc = fitz.open(PDF_PATH)

# 先观察前60页，因为管理层讨论、业务分析等主要章节集中在这里
for page_num in range(min(60, len(doc))):

    page = doc[page_num]
    text = page.get_text()

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    print("\n" + "=" * 70)
    print(f"PDF Page {page_num + 1}")
    print("=" * 70)

    # 每页先只显示前15行，方便观察标题结构
    for line in lines[:15]:
        print(line)
