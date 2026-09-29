import pymupdf
import json
import os
import re

PDF_PATH = "documents/人保财险2024年报.pdf"
OUTPUT_PATH = "data/chunks_v3.json"

COMPANY = "人保财险"
YEAR = 2024
SOURCE = "人保财险2024年报.pdf"

# 目标长度不是绝对限制。
# 普通 Chunk 尽量控制在这个范围附近。
TARGET_SIZE = 1000

# 超过这个长度时，必须进一步拆分。
MAX_SIZE = 1400

# 太短的文本块尽量和前后内容合并。
MIN_SIZE = 250


def clean_text(text):
    text = text.replace("\u00a0", " ")

    lines = [line.strip() for line in text.splitlines()]

    cleaned_lines = []

    for line in lines:

        # 1. 删除每页重复出现的公司名 + 年报页脚
        if "PICC Property and Casualty Company Limited" in line:
            continue

        # 2. 删除只有页码的独立行，例如：
        # 36
        # 116
        # 224
        if re.fullmatch(r"\d{1,3}", line):
            continue

        cleaned_lines.append(line)

    # 3. 清理连续空行
    final_lines = []
    previous_blank = False

    for line in cleaned_lines:
        if line == "":
            if not previous_blank:
                final_lines.append("")
            previous_blank = True
        else:
            final_lines.append(line)
            previous_blank = False

    return "\n".join(final_lines).strip()

def split_into_blocks(text):
    """
    根据 PDF 中的空行，把页面先拆成较自然的文本块。
    """

    blocks = re.split(r"\n\s*\n", text)

    return [
        block.strip()
        for block in blocks
        if block.strip()
    ]


def split_long_text(text, max_size=MAX_SIZE):
    """
    如果一个文本块本身过长，
    优先寻找句号、问号、分号等自然句子边界。
    """

    if len(text) <= max_size:
        return [text]

    # 尽量按英文句子边界切
    sentences = re.split(
        r"(?<=[.!?;])\s+",
        text
    )

    pieces = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # 当前内容还能容纳这个句子
        if len(current) + len(sentence) + 1 <= max_size:
            if current:
                current += " " + sentence
            else:
                current = sentence

        else:
            # 先保存已有内容
            if current:
                pieces.append(current)
                current = ""

            # 如果单个句子本身就超长，只能兜底硬切
            if len(sentence) > max_size:
                start = 0

                while start < len(sentence):
                    end = start + max_size
                    pieces.append(sentence[start:end].strip())
                    start = end

            else:
                current = sentence

    if current:
        pieces.append(current)

    return pieces


def make_page_chunks(text):
    """
    一页内部的核心 Chunking 逻辑：
    1. 先识别自然文本块
    2. 超长块进一步拆分
    3. 小块组合到目标长度附近
    """

    blocks = split_into_blocks(text)

    expanded_blocks = []

    # 先解决单个 block 过长的问题
    for block in blocks:
        expanded_blocks.extend(split_long_text(block))

    chunks = []
    current = ""

    for block in expanded_blocks:

        # 可以安全加入当前 Chunk
        if len(current) + len(block) + 2 <= TARGET_SIZE:
            if current:
                current += "\n\n" + block
            else:
                current = block

        else:
            # 当前 Chunk 已经比较完整，先保存
            if current:
                chunks.append(current)

            current = block

            # 极端情况下再次保证长度
            if len(current) > MAX_SIZE:
                long_parts = split_long_text(current)

                chunks.extend(long_parts[:-1])
                current = long_parts[-1]

    if current:
        # 最后一块过短时，尽量与上一块合并
        if (
            chunks
            and len(current) < MIN_SIZE
            and len(chunks[-1]) + len(current) + 2 <= MAX_SIZE
        ):
            chunks[-1] += "\n\n" + current
        else:
            chunks.append(current)

    return chunks


doc = pymupdf.open(PDF_PATH)

all_chunks = []

for page_index, page in enumerate(doc):
    page_number = page_index + 1

    raw_text = page.get_text()

    text = clean_text(raw_text)

    if not text:
        continue

    page_chunks = make_page_chunks(text)

    for chunk_index, chunk_text in enumerate(page_chunks):

        chunk_id = (
            f"picc_2024_p{page_number}_c{chunk_index + 1}"
        )

        # 给 Embedding 使用的文本增加必要上下文。
        embedding_text = (
            f"Company: {COMPANY}\n"
            f"Year: {YEAR}\n"
            f"PDF page: {page_number}\n\n"
            f"{chunk_text}"
        )

        chunk = {
            "id": chunk_id,
            "company": COMPANY,
            "year": YEAR,
            "page": page_number,
            "source": SOURCE,
            "text": chunk_text,
            "embedding_text": embedding_text,
            "char_count": len(chunk_text)
        }

        all_chunks.append(chunk)


os.makedirs("data", exist_ok=True)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )


# ---------- 质量检查 ----------

lengths = [
    chunk["char_count"]
    for chunk in all_chunks
]

print("V3 Chunk 制作完成！")
print("PDF页数：", len(doc))
print("Chunk总数：", len(all_chunks))
print("保存位置：", OUTPUT_PATH)

print("\n===== Chunk长度统计 =====")
print("最短：", min(lengths))
print("最长：", max(lengths))
print(
    "平均：",
    round(sum(lengths) / len(lengths), 1)
)

too_long = [
    chunk
    for chunk in all_chunks
    if chunk["char_count"] > MAX_SIZE
]

print(
    "超过最大长度的Chunk：",
    len(too_long)
)
