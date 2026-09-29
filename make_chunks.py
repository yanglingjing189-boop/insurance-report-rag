import pymupdf
import json
import os

PDF_PATH = "documents/人保财险2024年报.pdf"
OUTPUT_PATH = "data/chunks_v2.json"

COMPANY = "人保财险"
YEAR = 2024
SOURCE = "人保财险2024年报.pdf"

CHUNK_SIZE = 1000


def split_text(text, chunk_size=1000):
    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(current_chunk) + len(paragraph) <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


doc = pymupdf.open(PDF_PATH)

all_chunks = []

for page_index, page in enumerate(doc):
    page_number = page_index + 1

    text = page.get_text().strip()

    if not text:
        continue

    page_chunks = split_text(
        text,
        chunk_size=CHUNK_SIZE
    )

    for chunk_index, chunk_text in enumerate(page_chunks):
        chunk = {
            "id": f"picc_2024_p{page_number}_c{chunk_index + 1}",
            "company": COMPANY,
            "year": YEAR,
            "page": page_number,
            "source": SOURCE,
            "text": chunk_text
        }

        all_chunks.append(chunk)


os.makedirs("data", exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        all_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


print("V2 Chunk 制作完成！")
print("PDF页数：", len(doc))
print("Chunk总数：", len(all_chunks))
print("保存位置：", OUTPUT_PATH)
