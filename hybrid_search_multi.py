import json
import re

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/chunks_rag_multi.json"
EMBEDDINGS_PATH = "data/embeddings_multi.npy"

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

VECTOR_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60


# ============================================================
# 1. 公司名称与常见简称
# ============================================================

COMPANY_ALIASES = {
    "人保财险": [
        "人保财险",
        "中国人民财产保险",
        "中国人民财产保险股份有限公司",
        "PICC P&C",
    ],
    "中国人保": [
        "中国人保",
        "中国人民保险集团",
        "中国人民保险集团股份有限公司",
        "PICC Group",
    ],
    "中国平安": [
        "中国平安",
        "平安集团",
        "中国平安保险集团",
        "中国平安保险（集团）股份有限公司",
    ],
    "中国太保": [
        "中国太保",
        "太保集团",
        "中国太平洋保险集团",
        "中国太平洋保险（集团）股份有限公司",
    ],
    "中国太保财险": [
        "中国太保财险",
        "太保财险",
        "中国太平洋财产保险",
        "中国太平洋财产保险股份有限公司",
    ],
    "中国太平": [
        "中国太平",
        "中国太平保险集团",
        "中国太平保险集团有限责任公司",
    ],
    "中国人寿": [
        "中国人寿",
        "中国人寿保险集团",
        "中国人寿保险（集团）公司",
    ],
    "新华人寿": [
        "新华人寿",
        "新华保险",
        "新华人寿保险股份有限公司",
    ],
    "阳光保险": [
        "阳光保险",
        "阳光保险集团",
        "阳光保险集团股份有限公司",
    ],
    "中国再保险": [
        "中国再保险",
        "中再集团",
        "中国再保险集团",
        "中国再保险（集团）股份有限公司",
    ],
}


def detect_companies(query):
    """
    从问题中识别公司。

    关键规则：
    如果已经识别到“中国太保财险”，
    就不再把其中的“中国太保”重复识别为另一家公司。
    """

    query_lower = query.lower()

    alias_pairs = []

    for company, aliases in COMPANY_ALIASES.items():
        for alias in aliases:
            alias_pairs.append(
                (company, alias)
            )

    # 长别名优先
    alias_pairs.sort(
        key=lambda x: len(x[1]),
        reverse=True
    )

    detected = []
    matched_spans = []

    for company, alias in alias_pairs:

        alias_lower = alias.lower()

        start = 0

        while True:

            position = query_lower.find(
                alias_lower,
                start
            )

            if position == -1:
                break

            end = position + len(alias_lower)

            # 如果这个字符区间已经被一个更长的公司名称覆盖，
            # 就不重复识别。
            overlaps = any(
                position >= old_start
                and end <= old_end
                for old_start, old_end
                in matched_spans
            )

            if not overlaps:

                if company not in detected:
                    detected.append(company)

                matched_spans.append(
                    (position, end)
                )

            start = position + 1

    return detected


# ============================================================
# 2. 中英文分词
# ============================================================

def tokenize(text):

    text = str(text).lower()

    raw_tokens = jieba.lcut(text)

    tokens = []

    for token in raw_tokens:

        token = token.strip()

        if not token:
            continue

        parts = re.findall(
            r"[\u4e00-\u9fff]+|[a-z]+|\d+(?:\.\d+)?",
            token
        )

        tokens.extend(parts)

    return tokens


# ============================================================
# 3. 加载数据
# ============================================================

def load_data():

    print("正在加载 Chunk...")

    with open(
        CHUNKS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    print("正在加载 Embedding...")

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Chunk 数量与 Embedding 数量不一致"
        )

    print(f"Chunk 数量：{len(chunks)}")
    print(
        f"Embedding shape："
        f"{embeddings.shape}"
    )

    return chunks, embeddings


# ============================================================
# 4. 根据公司过滤搜索范围
# ============================================================

def get_candidate_indices(
    query,
    chunks
):

    companies = detect_companies(query)

    if not companies:

        indices = list(
            range(len(chunks))
        )

        return indices, companies

    indices = [
        i
        for i, chunk in enumerate(chunks)
        if chunk["company"] in companies
    ]

    return indices, companies


# ============================================================
# 5. BM25
# ============================================================

def build_bm25(
    chunks,
    candidate_indices
):

    corpus = [
        tokenize(
            chunks[i]["embedding_text"]
        )
        for i in candidate_indices
    ]

    bm25 = BM25Okapi(corpus)

    return bm25


# ============================================================
# 6. Vector Search
# ============================================================

def vector_search(
    query,
    model,
    chunks,
    embeddings,
    candidate_indices,
    top_k=VECTOR_TOP_K
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    candidate_embeddings = embeddings[
        candidate_indices
    ]

    scores = (
        candidate_embeddings
        @ query_embedding
    )

    local_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, local_index in enumerate(
        local_indices,
        start=1
    ):

        global_index = candidate_indices[
            local_index
        ]

        results.append({
            "index": int(global_index),
            "rank": rank,
            "score": float(
                scores[local_index]
            ),
            "chunk": chunks[
                global_index
            ],
        })

    return results


# ============================================================
# 7. BM25 Search
# ============================================================

def bm25_search(
    query,
    bm25,
    chunks,
    candidate_indices,
    top_k=BM25_TOP_K
):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    local_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, local_index in enumerate(
        local_indices,
        start=1
    ):

        global_index = candidate_indices[
            local_index
        ]

        results.append({
            "index": int(global_index),
            "rank": rank,
            "score": float(
                scores[local_index]
            ),
            "chunk": chunks[
                global_index
            ],
        })

    return results


# ============================================================
# 8. RRF 融合
# ============================================================

def rrf_fusion(
    vector_results,
    bm25_results,
    top_k=10
):

    fused = {}

    for result in vector_results:

        index = result["index"]

        if index not in fused:
            fused[index] = {
                "index": index,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
                "chunk": result["chunk"],
            }

        fused[index]["vector_rank"] = (
            result["rank"]
        )

        fused[index]["rrf_score"] += (
            1.0 /
            (RRF_K + result["rank"])
        )

    for result in bm25_results:

        index = result["index"]

        if index not in fused:
            fused[index] = {
                "index": index,
                "rrf_score": 0.0,
                "vector_rank": None,
                "bm25_rank": None,
                "chunk": result["chunk"],
            }

        fused[index]["bm25_rank"] = (
            result["rank"]
        )

        fused[index]["rrf_score"] += (
            1.0 /
            (RRF_K + result["rank"])
        )

    ranked = sorted(
        fused.values(),
        key=lambda x: x["rrf_score"],
        reverse=True
    )

    return ranked[:top_k]


# ============================================================
# 9. 输出结果
# ============================================================

def print_results(
    title,
    results
):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):

        chunk = result["chunk"]

        print(f"\n[{i}] {chunk['id']}")

        print(
            f"公司：{chunk['company']} | "
            f"类型：{chunk['type']} | "
            f"章节：{chunk['section']} | "
            f"PDF页码：{chunk['page']}"
        )

        if "rrf_score" in result:

            print(
                f"RRF：{result['rrf_score']:.6f} | "
                f"Vector rank：{result['vector_rank']} | "
                f"BM25 rank：{result['bm25_rank']}"
            )

        else:

            print(
                f"Rank：{result['rank']} | "
                f"Score：{result['score']:.6f}"
            )

        print("内容：")

        print(
            chunk["text"][:500]
        )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    chunks, embeddings = load_data()

    print(
        "\n正在加载 Embedding 模型..."
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="mps"
    )

    query = input(
        "\n请输入问题："
    ).strip()

    if not query:
        print("问题不能为空")
        return

    # ---------- 公司识别 ----------

    candidate_indices, companies = (
        get_candidate_indices(
            query,
            chunks
        )
    )

    print("\n识别到的公司：")

    if companies:
        print(
            "、".join(companies)
        )
    else:
        print(
            "未指定公司 → 搜索全部公司"
        )

    print(
        f"候选 Chunk 数量："
        f"{len(candidate_indices)}"
    )

    # ---------- BM25 ----------

    print(
        "\n正在建立 BM25 索引..."
    )

    bm25 = build_bm25(
        chunks,
        candidate_indices
    )

    print(
        "BM25 索引建立完成"
    )

    print(
        "\nBM25 分词：",
        tokenize(query)
    )

    # ---------- Vector ----------

    vector_results = vector_search(
        query,
        model,
        chunks,
        embeddings,
        candidate_indices
    )

    # ---------- BM25 ----------

    bm25_results = bm25_search(
        query,
        bm25,
        chunks,
        candidate_indices
    )

    # ---------- RRF ----------

    hybrid_results = rrf_fusion(
        vector_results,
        bm25_results,
        top_k=10
    )

    # ---------- 输出 ----------

    print_results(
        "Vector Top 5",
        vector_results[:5]
    )

    print_results(
        "BM25 Top 5",
        bm25_results[:5]
    )

    print_results(
        "Hybrid RRF Top 10",
        hybrid_results
    )


if __name__ == "__main__":
    main()
