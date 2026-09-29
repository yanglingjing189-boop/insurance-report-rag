from pathlib import Path
import pymupdf

REPORT_DIR = Path("documents/annual_reports")

pdf_files = sorted(REPORT_DIR.glob("*.pdf"))

print(f"发现 {len(pdf_files)} 份 PDF\n")

for pdf_path in pdf_files:
    print("=" * 70)
    print(f"文件：{pdf_path.name}")

    try:
        doc = pymupdf.open(pdf_path)

        print(f"页数：{len(doc)}")

        first_page_text = doc[0].get_text("text").strip()

        if first_page_text:
            preview = first_page_text[:200].replace("\n", " | ")
            print("首页文字提取：成功")
            print(f"首页预览：{preview}")
        else:
            print("首页文字提取：失败（首页没有提取到文字）")

        doc.close()

    except Exception as e:
        print(f"PDF 打开失败：{e}")
