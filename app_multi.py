import streamlit as st
from sentence_transformers import SentenceTransformer

from rag_qa_multi_v2 import (
    load_data,
    run_rag,
    MODEL_NAME,
)


# ============================================================
# 页面设置
# ============================================================

st.set_page_config(
    page_title="保险公司财报 RAG 问答系统",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# 页面标题
# ============================================================

st.title("📊 保险公司财报 RAG 问答系统")

st.caption(
    "基于 10 家保险公司 2025 年度报告构建 · "
    "Vector + BM25 + RRF 混合检索 · "
    "DeepSeek 生成答案"
)

st.info(
    "输入有关保险公司 2025 年度报告的问题。"
    "系统会先从财报知识库中检索相关证据，"
    "再根据证据生成答案，并展示公司、章节、PDF 页码和 Chunk ID。"
)


# ============================================================
# 缓存加载知识库
# ============================================================

@st.cache_resource
def load_rag_system():

    chunks, embeddings = load_data()

    embedding_model = SentenceTransformer(
        MODEL_NAME,
        device="mps"
    )

    return (
        chunks,
        embeddings,
        embedding_model,
    )


with st.spinner("正在加载财报知识库和 Embedding 模型..."):

    chunks, embeddings, embedding_model = (
        load_rag_system()
    )


# ============================================================
# 知识库概览
# ============================================================

companies = sorted(
    set(
        chunk["company"]
        for chunk in chunks
    )
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "覆盖公司",
        f"{len(companies)} 家"
    )

with col2:
    st.metric(
        "知识库 Chunk",
        f"{len(chunks):,}"
    )

with col3:
    st.metric(
        "检索方式",
        "Vector + BM25"
    )


with st.expander(
    "查看知识库覆盖的公司"
):

    st.write(
        "、".join(companies)
    )


st.divider()


# ============================================================
# 示例问题
# ============================================================

st.subheader("💬 财报问答")

st.caption(
    "例如："
    "“人保财险2025年机动车辆险的综合成本率是多少？” "
    "或 "
    "“比较人保财险和中国太保财险2025年机动车辆险的综合成本率。”"
)


# ============================================================
# 输入问题
# ============================================================

query = st.text_area(
    "请输入问题",
    height=100,
    placeholder="请输入你想查询的保险公司财报问题……",
)


ask_button = st.button(
    "开始提问",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 执行 RAG
# ============================================================

if ask_button:

    if not query.strip():

        st.warning(
            "请先输入一个问题。"
        )

    else:

        try:

            with st.spinner(
                "正在检索财报并生成答案..."
            ):

                result = run_rag(
                    query.strip(),
                    chunks,
                    embeddings,
                    embedding_model,
                )


            # ====================================================
            # 检索信息
            # ====================================================

            st.success(
                "问答完成"
            )

            info_col1, info_col2 = (
                st.columns(2)
            )

            with info_col1:

                if result.get("companies", []):

                    st.metric(
                        "识别公司",
                        "、".join(
                            result.get("companies", [])
                        )
                    )

                else:

                    st.metric(
                        "识别公司",
                        "全部公司"
                    )

            with info_col2:

                st.metric(
                    "候选 Chunk",
                    result["candidate_count"]
                )


            # ====================================================
            # 最终答案
            # ====================================================

            st.subheader(
                "🤖 AI 回答"
            )

            st.markdown(
                result["answer"]
            )


            # ====================================================
            # 召回证据
            # ====================================================

            st.subheader(
                "🔎 召回证据"
            )

            st.caption(
                "以下内容是 Vector + BM25 "
                "经过 RRF 融合后提供给大模型的证据。"
            )


            for i, item in enumerate(
                result["hybrid_results"],
                start=1
            ):

                chunk = item["chunk"]

                title = (
                    f"来源{i}｜"
                    f"{chunk['company']}｜"
                    f"PDF 第 {chunk['page']} 页｜"
                    f"{chunk['type']}"
                )

                with st.expander(
                    title,
                    expanded=(i <= 2)
                ):

                    meta_col1, meta_col2 = (
                        st.columns(2)
                    )

                    with meta_col1:

                        st.write(
                            "**公司：**",
                            chunk["company"]
                        )

                        st.write(
                            "**年份：**",
                            chunk["year"]
                        )

                        st.write(
                            "**PDF 页码：**",
                            chunk["page"]
                        )

                    with meta_col2:

                        st.write(
                            "**章节：**",
                            chunk["section"]
                        )

                        st.write(
                            "**类型：**",
                            chunk["type"]
                        )

                        st.write(
                            "**Chunk ID：**",
                            chunk["id"]
                        )


                    st.write(
                        "**RRF 分数：**",
                        f"{item['rrf_score']:.6f}"
                    )

                    st.markdown(
                        "**原始证据内容：**"
                    )

                    st.code(
                        chunk["text"],
                        language=None
                    )


            # ====================================================
            # 系统说明
            # ====================================================

            with st.expander(
                "查看本次 RAG 流程"
            ):

                st.markdown(
                    """
**本次问题的处理流程：**

1. 从问题中识别保险公司；
2. 根据公司 Metadata 缩小检索范围；
3. 使用 Embedding 进行 Vector Search；
4. 使用 BM25 进行关键词检索；
5. 使用 RRF 融合两种检索结果；
6. 选取排名靠前的证据 Chunk；
7. 将证据交给 DeepSeek；
8. DeepSeek 仅根据检索证据生成答案并标注出处。
"""
                )


        except Exception as e:

            st.error(
                "运行过程中出现错误："
            )

            st.exception(e)


# ============================================================
# 页脚
# ============================================================

st.divider()

st.caption(
    "课程项目：人工智能与数据分析｜"
    "保险公司年度报告 RAG 知识库"
)
