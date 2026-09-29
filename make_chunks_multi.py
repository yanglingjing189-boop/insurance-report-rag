import os
import re
import json
import pymupdf
from pathlib import Path

REPORT_DIR = "documents/annual_reports"
OUTPUT_PATH = "data/chunks_multi.json"

TARGET = 1000
MAX_LEN = 1400
MIN_LEN = 250


# ============================================================
# 3份书签异常报告：使用正文一级标题
# ============================================================

FALLBACK_SECTIONS = {
    "人保财险": [
        "一、公司简介",
        "二、保险责任准备金信息",
        "三、风险管理状况信息",
        "四、保险产品经营信息",
        "五、偿付能力信息",
        "六、关联交易总体情况",
        "七、消费者权益保护信息",
        "八、公司治理信息",
        "九、重大事项信息",
        "十、财务会计信息及财务会计报告",
    ],

    "中国人寿": [
        "一、公司概况",
        "二、公司治理信息",
        "三、财务会计信息",
        "四、保险责任准备金信息",
        "五、风险管理状况信息",
        "六、保险产品经营信息",
        "七、偿付能力信息",
        "八、重大内部交易信息",
        "九、关联交易总体情况",
        "十、重大事项信息",
        "十一、消费者权益保护信息",
    ],

    "中国太保财险": [
        "一、公司简介",
        "二、公司治理信息",
        "三、重大事项信息",
        "四、财务会计信息",
        "五、保险责任准备金信息",
        "六、风险管理状况信息",
        "七、保险产品经营信息",
        "八、偿付能力信息",
        "九、关联交易总体情况",
        "十、消费者保护重大工作信息",
        "十一、其他信息",
    ],
}


def clean_text(text):
    """
    基础文本清洗。
    不做过度清洗，避免误删公司名称、章节标题和有意义的短文本。
    """
    text = text.replace("\u00a0", " ")

    lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            lines.append("")
            continue

        # 删除单独出现的页码，例如：
        # 1
        # 23
        # 105
        if re.fullmatch(r"\d{1,3}", line):
            continue

        lines.append(line)

    text = "\n".join(lines)

    # 压缩过多空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_long_text(text, max_len=MAX_LEN):
    """
    将超长文本按中英文句号、问号、感叹号、分号等切分。
    """
    if len(text) <= max_len:
        return [text]

    sentences = re.split(
        r"(?<=[。！？；.!?;])\s*",
        text
    )

    pieces = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        # 单句本身已经超过 MAX_LEN
        if len(sentence) > max_len:
            if current:
                pieces.append(current.strip())
                current = ""

            for i in range(0, len(sentence), max_len):
                pieces.append(sentence[i:i + max_len].strip())

            continue

        if not current:
            current = sentence

        elif len(current) + len(sentence) + 1 <= max_len:
            current += " " + sentence

        else:
            pieces.append(current.strip())
            current = sentence

    if current:
        pieces.append(current.strip())

    return [x for x in pieces if x]


def make_chunks_from_text(text):
    """
    将一段文本切成约 TARGET 字符的 chunk。
    """
    text = clean_text(text)

    if not text:
        return []

    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", text)
        if block.strip()
    ]

    expanded_blocks = []

    for block in blocks:
        if len(block) > MAX_LEN:
            expanded_blocks.extend(split_long_text(block))
        else:
            expanded_blocks.append(block)

    chunks = []
    current = ""

    for block in expanded_blocks:

        if not current:
            current = block
            continue

        candidate = current + "\n\n" + block

        if len(candidate) <= TARGET:
            current = candidate

        else:
            chunks.append(current.strip())
            current = block

    if current:
        chunks.append(current.strip())

    # 如果仍有超长 chunk，再强制切一次
    final_chunks = []

    for chunk in chunks:
        if len(chunk) > MAX_LEN:
            final_chunks.extend(split_long_text(chunk))
        else:
            final_chunks.append(chunk)

    return [x for x in final_chunks if x.strip()]


def parse_filename(path):
    """
    例如：
    中国平安_2025_年报.pdf
        ↓
    company = 中国平安
    year = 2025
    """
    name = Path(path).name

    match = re.match(
        r"(.+?)_(\d{4})_年报\.pdf$",
        name
    )

    if not match:
        raise ValueError(
            f"文件名不符合规则：{name}"
        )

    company = match.group(1)
    year = match.group(2)

    return company, year


# ============================================================
# 正常 PDF：使用一级书签
# ============================================================

def build_bookmark_sections(doc):
    """
    读取 PDF 一级书签。
    返回：
    [
        (起始PDF页码, section名称),
        ...
    ]
    """
    toc = doc.get_toc()

    sections = []

    for item in toc:
        level, title, page = item

        if level == 1:
            title = title.strip()

            if title:
                sections.append(
                    (page, title)
                )

    sections.sort(key=lambda x: x[0])

    return sections


def get_bookmark_section(page_number, sections):
    """
    根据 PDF 页码判断当前 section。
    """
    current_section = "Unknown"

    for start_page, title in sections:

        if page_number >= start_page:
            current_section = title

        else:
            break

    return current_section


# ============================================================
# 异常 PDF：按正文一级标题进行页内切分
# ============================================================

def split_page_by_fallback_sections(
    text,
    current_section,
    section_titles
):
    """
    在一页内部寻找“整行等于一级标题”的位置。

    这样可以处理：
    人保财险 PDF 第3页：
        一、公司简介
        ...
        二、保险责任准备金信息

    即同一页存在两个一级章节的情况。

    返回：
    segments = [
        (section, text),
        ...
    ]

    以及本页结束时的 current_section。
    """

    raw_lines = text.splitlines()

    segments = []

    buffer = []
    section = current_section

    title_set = set(section_titles)

    for raw_line in raw_lines:

        stripped = raw_line.strip()

        # 必须整行等于章节标题
        # 避免：
        # “详见本报告‘八、公司治理信息’”
        # 被错误识别成新章节
        if stripped in title_set:

            # 先保存前一个 section 的内容
            if buffer:
                segment_text = "\n".join(buffer).strip()

                if segment_text:
                    segments.append(
                        (section, segment_text)
                    )

                buffer = []

            # 从这里开始进入新的 section
            section = stripped

            # 标题本身保留在正文里
            buffer.append(stripped)

        else:
            buffer.append(raw_line)

    if buffer:
        segment_text = "\n".join(buffer).strip()

        if segment_text:
            segments.append(
                (section, segment_text)
            )

    return segments, section


# ============================================================
# 单份 PDF 处理
# ============================================================

def process_pdf(path):

    company, year = parse_filename(path)

    print()
    print("=" * 70)
    print(f"正在处理：{company} {year}")
    print("=" * 70)

    doc = pymupdf.open(path)

    use_fallback = company in FALLBACK_SECTIONS

    if use_fallback:

        print("章节模式：正文一级标题 fallback")

        section_titles = FALLBACK_SECTIONS[company]

        bookmark_sections = None

    else:

        print("章节模式：PDF 一级书签")

        bookmark_sections = build_bookmark_sections(doc)

        print(
            f"检测到 {len(bookmark_sections)} 个一级书签"
        )

        section_titles = None

    chunks = []

    chunk_counter_by_page = {}

    # fallback 模式需要跨页记住当前章节
    current_section = "Unknown"

    for page_index in range(len(doc)):

        page_number = page_index + 1

        page = doc[page_index]

        raw_text = page.get_text("text")

        if not raw_text.strip():
            continue

        # ----------------------------------------------------
        # fallback 报告
        # ----------------------------------------------------

        if use_fallback:

            segments, current_section = \
                split_page_by_fallback_sections(
                    raw_text,
                    current_section,
                    section_titles
                )

        # ----------------------------------------------------
        # 正常书签报告
        # ----------------------------------------------------

        else:

            section = get_bookmark_section(
                page_number,
                bookmark_sections
            )

            segments = [
                (section, raw_text)
            ]

        # ----------------------------------------------------
        # 对每个 section segment 独立 chunk
        # ----------------------------------------------------

        for section, segment_text in segments:

            page_chunks = make_chunks_from_text(
                segment_text
            )

            for chunk_text in page_chunks:

                chunk_counter_by_page.setdefault(
                    page_number,
                    0
                )

                chunk_counter_by_page[page_number] += 1

                chunk_number = \
                    chunk_counter_by_page[page_number]

                chunk_id = (
                    f"{company}_{year}_"
                    f"p{page_number}_"
                    f"c{chunk_number}"
                )

                embedding_text = (
                    f"公司：{company}\n"
                    f"年份：{year}\n"
                    f"PDF页码：{page_number}\n"
                    f"章节：{section}\n\n"
                    f"{chunk_text}"
                )

                chunk = {
                    "id": chunk_id,
                    "company": company,
                    "year": year,
                    "section": section,
                    "page": page_number,
                    "source": Path(path).name,
                    "text": chunk_text,
                    "embedding_text": embedding_text,
                    "char_count": len(chunk_text),
                }

                chunks.append(chunk)

    doc.close()

    print(
        f"生成 Chunk：{len(chunks)}"
    )

    unknown_count = sum(
        1
        for c in chunks
        if c["section"] == "Unknown"
    )

    print(
        f"Unknown Chunk：{unknown_count}"
    )

    return chunks


# ============================================================
# 主程序
# ============================================================

def main():

    report_paths = sorted(
        Path(REPORT_DIR).glob("*.pdf")
    )

    if not report_paths:
        raise FileNotFoundError(
            f"没有在 {REPORT_DIR} 找到 PDF"
        )

    print(
        f"共发现 {len(report_paths)} 份报告"
    )

    all_chunks = []

    company_stats = {}

    for path in report_paths:

        chunks = process_pdf(str(path))

        all_chunks.extend(chunks)

        if chunks:
            company = chunks[0]["company"]
            company_stats[company] = len(chunks)

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

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

    # --------------------------------------------------------
    # 统计
    # --------------------------------------------------------

    lengths = [
        c["char_count"]
        for c in all_chunks
    ]

    ids = [
        c["id"]
        for c in all_chunks
    ]

    duplicate_ids = (
        len(ids) - len(set(ids))
    )

    print()
    print("=" * 70)
    print("全部处理完成")
    print("=" * 70)

    print(
        f"报告数量：{len(report_paths)}"
    )

    print(
        f"Chunk 总数：{len(all_chunks)}"
    )

    print(
        f"已保存：{OUTPUT_PATH}"
    )

    print()
    print("各公司 Chunk 数：")

    for company, count in company_stats.items():
        print(
            f"  {company}: {count}"
        )

    if lengths:

        print()
        print("Chunk 长度统计：")

        print(
            f"  最短：{min(lengths)}"
        )

        print(
            f"  最长：{max(lengths)}"
        )

        print(
            f"  平均：{sum(lengths) / len(lengths):.1f}"
        )

        over_max = sum(
            1
            for x in lengths
            if x > MAX_LEN
        )

        print(
            f"  超过 MAX_LEN：{over_max}"
        )

    print()
    print(
        f"重复 Chunk ID：{duplicate_ids}"
    )


if __name__ == "__main__":
    main()
