import streamlit as st

from rag_qa import hybrid_search, generate_answer


# =========================
# 页面设置
# =========================

st.set_page_config(
    page_title="保险财报 RAG 问答助手",
    page_icon="📊",
    layout="wide"
)


# =========================
# 页面标题
# =========================

st.title("📊 保险财报 RAG 问答助手")

st.write(
    "基于保险公司年报，使用 Vector + BM25 + RRF "
    "进行混合检索，并由 LLM 根据检索证据生成答案。"
)

st.info(
    "当前原型知识库：人保财险 2024 年年度报告"
)


# =========================
# 用户输入
# =========================

question = st.text_area(
    "请输入你的问题：",
    placeholder="例如：Why did PICC's investment income increase in 2024?",
    height=100
)


# =========================
# 问答
# =========================

if st.button("开始问答", type="primary"):

    if not question.strip():

        st.warning("请先输入一个问题。")

    else:

        with st.spinner("正在检索年报证据..."):

            search_results = hybrid_search(
                question,
                top_k=5
            )

        with st.spinner("正在根据证据生成答案..."):

            answer = generate_answer(
                question,
                search_results
            )

        st.subheader("回答")

        st.write(answer)

        st.subheader("检索证据")

        for i, result in enumerate(
            search_results,
            start=1
        ):

            chunk = result["chunk"]

            with st.expander(
                f"Source {i}｜"
                f"{chunk['company']}｜"
                f"第 {chunk['page']} 页"
            ):

                st.write(
                    f"**章节：** {chunk['section']}"
                )

                st.write(
                    f"**Chunk ID：** {chunk['id']}"
                )

                st.write(
                    f"**RRF Score：** "
                    f"{result['rrf_score']:.6f}"
                )

                st.write("**原文：**")

                st.write(chunk["text"])
