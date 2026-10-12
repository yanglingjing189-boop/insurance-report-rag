import os

from openai import OpenAI

from hybrid_search_multi import (
    load_data,
    get_candidate_indices,
    build_bm25,
    vector_search,
    bm25_search,
    rrf_fusion,
    tokenize,
    MODEL_NAME,
)

from sentence_transformers import SentenceTransformer


# ============================================================
# 基本设置
# ============================================================

TOP_K = 6

DEEPSEEK_MODEL = "deepseek-flash"


# ============================================================
# 构造给 LLM 的证据
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]

        source = (
            f"[来源{i}]\n"
            f"公司：{chunk['company']}\n"
            f"年份：{chunk['year']}\n"
            f"章节：{chunk['section']}\n"
            f"PDF页码：{chunk['page']}\n"
            f"类型：{chunk['type']}\n"
            f"Chunk ID：{chunk['id']}\n"
            f"内容：\n"
            f"{chunk['text']}"
        )

        context_parts.append(source)

    return "\n\n".join(context_parts)


# ============================================================
# 调用 DeepSeek
# ============================================================

def ask_llm(
    query,
    context
):

    api_key = os.getenv(
        "DEEPSEEK_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "没有找到环境变量 DEEPSEEK_API_KEY"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    system_prompt = """
你是一名严谨的保险行业财报研究助手。

你只能根据用户提供给你的“检索证据”回答问题。

必须遵守以下规则：

1. 不得使用检索证据之外的事实补充答案。
2. 如果证据不足以回答问题，明确说明“当前检索证据不足以回答该问题”。
3. 涉及数字时，必须严格按照证据中的数字回答，不得自行猜测。
4. 如果问题涉及多家公司，必须分别说明每家公司的结果。
5. 如果不同公司的指标名称或统计口径不同，应明确提醒用户，不要擅自认为完全可比。
6. 每一个关键结论后必须标注来源，格式为：
   [来源1]
   [来源2]
7. 最后单独输出“出处”部分。
8. 出处必须包含：
   公司、年份、章节、PDF页码、Chunk ID。
9. 回答使用中文，尽量简洁、清楚。
"""

    user_prompt = f"""
用户问题：

{query}


以下是检索证据：

{context}


请严格根据以上证据回答用户问题。
"""

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0,
    )

    return (
        response
        .choices[0]
        .message
        .content
    )



def rerank_profit_evidence(results):
    """对归母净利润证据进行辅助重排序，不预设正确 Chunk ID。"""

    phrases = [
        "归属于母公司股东的净利润",
        "归属于母公司所有者的净利润",
        "归属于本公司股东的净利润",
        "归属于公司股东的净利润",
        "归属于母公司普通股股东的净利润",
        "profit attributable to owners",
        "profit attributable to shareholders",
        "归母净利润",
    ]

    def score(result):
        chunk = result["chunk"]
        text = chunk.get("text", "").lower()
        hits = sum(phrase.lower() in text for phrase in phrases)
        year_hit = "2025" in text
        return (
            hits > 0,
            hits,
            year_hit,
            result["rrf_score"],
        )

    return sorted(results, key=score, reverse=True)


# ============================================================
# RAG 主流程
# ============================================================


def run_rag(
    query,
    chunks,
    embeddings,
    embedding_model
):
    # 1. 识别问题涉及的公司
    candidate_indices, companies = get_candidate_indices(
        query, chunks
    )

    # 2. 判断是否为10家公司全景问题
    all_companies = list(dict.fromkeys(
        chunk["company"] for chunk in chunks
    ))

    panoramic_keywords = [
        "这10家", "这十家", "全部10家", "全部十家",
        "所有公司", "所有保险公司", "各家公司",
        "10家公司", "十家公司"
    ]

    is_panoramic = any(
        keyword in query
        for keyword in panoramic_keywords
    )

    if is_panoramic and not companies:
        companies = all_companies
        candidate_indices = list(range(len(chunks)))

    # 3. 为归母净利润全景问题使用精准检索查询
    retrieval_query = query

    if is_panoramic and (
        "归属于母公司股东的净利润" in query
        or "归母净利润" in query
    ):
        retrieval_query = (
            "2025年归属于母公司股东的净利润是多少？"
        )

    # 4. 封装单次混合检索
    def search_one_company(indices, top_k):
        bm25 = build_bm25(chunks, indices)

        vector_results = vector_search(
            retrieval_query,
            embedding_model,
            chunks,
            embeddings,
            indices
        )

        bm25_results = bm25_search(
            retrieval_query,
            bm25,
            chunks,
            indices
        )

        hybrid_results = rrf_fusion(
            vector_results,
            bm25_results,
            top_k=top_k
        )

        return vector_results, bm25_results, hybrid_results

    # 4. 单公司或未指定公司：保持原来的检索逻辑
    if len(companies) <= 1:
        vector_results, bm25_results, hybrid_results = (
            search_one_company(candidate_indices, TOP_K)
        )

    # 5. 多公司：逐家公司独立检索
    else:
        vector_results = []
        bm25_results = []
        hybrid_results = []

        # 每家公司保留最多3条证据
        per_company_top_k = 3

        for company in companies:
            company_indices = [
                i for i in candidate_indices
                if chunks[i]["company"] == company
            ]

            if not company_indices:
                continue

            v_results, b_results, h_results = (
                search_one_company(
                    company_indices,
                    15 if is_panoramic else per_company_top_k
                )
            )

            if is_panoramic and (
                "归属于母公司股东的净利润" in query
                or "归母净利润" in query
            ):
                h_results = rerank_profit_evidence(h_results)
                h_results = h_results[:per_company_top_k]

            vector_results.extend(v_results)
            bm25_results.extend(b_results)
            hybrid_results.extend(h_results)

    # 6. 构造证据上下文
    context = build_context(hybrid_results)

    # 7. 调用 DeepSeek
    answer = ask_llm(query, context)

    return {
        "query": query,
        "companies": companies,
        "candidate_count": len(candidate_indices),
        "vector_results": vector_results,
        "bm25_results": bm25_results,
        "hybrid_results": hybrid_results,
        "context": context,
        "answer": answer,
    }



# ============================================================
# 显示召回证据
# ============================================================

def print_retrieval_results(
    results
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Hybrid RRF 召回证据"
    )

    print(
        "=" * 70
    )

    for i, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]

        print(
            f"\n[来源{i}] "
            f"{chunk['id']}"
        )

        print(
            f"公司：{chunk['company']}"
        )

        print(
            f"章节：{chunk['section']}"
        )

        print(
            f"PDF页码：{chunk['page']}"
        )

        print(
            f"类型：{chunk['type']}"
        )

        print(
            f"RRF："
            f"{result['rrf_score']:.6f}"
        )

        print(
            "内容："
        )

        print(
            chunk["text"][:700]
        )


# ============================================================
# 主程序
# ============================================================

def main():

    print(
        "正在加载知识库..."
    )

    chunks, embeddings = (
        load_data()
    )

    print(
        "\n正在加载 Embedding 模型..."
    )

    embedding_model = (
        SentenceTransformer(
            MODEL_NAME,
            device="mps"
        )
    )

    query = input(
        "\n请输入问题："
    ).strip()

    if not query:
        print(
            "问题不能为空"
        )
        return

    print(
        "\nBM25 分词：",
        tokenize(query)
    )

    print(
        "\n正在进行 RAG 问答..."
    )

    result = run_rag(
        query,
        chunks,
        embeddings,
        embedding_model
    )

    print(
        "\n识别到的公司："
    )

    if result["companies"]:
        print(
            "、".join(
                result["companies"]
            )
        )
    else:
        print(
            "未指定公司 → 搜索全部公司"
        )

    print(
        f"候选 Chunk 数量："
        f"{result['candidate_count']}"
    )

    print_retrieval_results(
        result["hybrid_results"]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DeepSeek 最终回答"
    )

    print(
        "=" * 70
    )

    print(
        result["answer"]
    )


if __name__ == "__main__":
    main()
