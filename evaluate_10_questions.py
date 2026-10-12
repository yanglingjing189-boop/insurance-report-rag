import json
import os
from datetime import datetime
from pathlib import Path

import rag_qa_multi_v2 as rag

QUESTIONS = [
    ("Q1", "比较人保财险和中国太保财险2025年机动车辆险的综合成本率。",
     ["人保财险_2025_p10_table1", "中国太保财险_2025_p39_c2"]),
    ("Q2", "人保财险2025年责任险的综合成本率是多少？",
     ["人保财险_2025_p10_table1"]),
    ("Q3", "中国太保财险2025年责任险的承保综合成本率是多少？",
     ["中国太保财险_2025_p39_c2"]),
    ("Q4", "中国太保财险2025年机动车辆保险分部主要包括什么业务？",
     ["中国太保财险_2025_p184_c1"]),
    ("Q5", "人保财险2025年机动车辆险的保险服务收入是多少？",
     ["人保财险_2025_p10_table1", "人保财险_2025_p162_c1"]),
    ("Q6", "根据人保财险2025年的数据，机动车辆险的承保利润占保险服务收入的比例约为多少？",
     ["人保财险_2025_p10_table1"]),
    ("Q7", "比较人保财险和中国太保财险2025年责任险的综合成本率，并说明哪家公司数值更高、相差多少个百分点。",
     ["人保财险_2025_p10_table1", "中国太保财险_2025_p39_c2"]),
    ("Q8", "中国太保财险2025年人民币保单使用的即期折现率假设范围是多少？",
     ["中国太保财险_2025_p116_c1"]),
    ("Q9", "中国平安2025年归属于母公司股东的净利润是多少？",
     ["中国平安_2025_p83_c1"]),
    ("Q10", "在这10家保险公司中，比较2025年归属于母公司股东的净利润，并列出检索证据能够支持的公司及对应数值。",
     [
         "中国人保_2025_p8_c2",
         "中国人寿_2025_p26_c1",
         "中国再保险_2025_p22_c2",
         "中国太保_2025_p87_c1",
         "中国太保财险_2025_p18_c2",
         "中国太平_2025_p27_c1",
         "中国平安_2025_p83_c1",
         "人保财险_2025_p41_c1",
         "新华人寿_2025_p28_c1",
         "阳光保险_2025_p225_c1",
     ]),
]

def main():
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("未检测到 DEEPSEEK_API_KEY，停止测试。")
        return

    print("加载知识库与 Embedding 模型……", flush=True)
    chunks, embeddings = rag.load_data()
    model = rag.SentenceTransformer(rag.MODEL_NAME, device="mps")

    output = Path("evaluation_after_v2.json")
    records = []

    for number, (qid, question, gold_ids) in enumerate(QUESTIONS, 1):
        print(f"\n[{number}/10] 正在测试 {qid}……", flush=True)
        record = {
            "id": qid,
            "question": question,
            "reference_chunk_ids": gold_ids,
            "status": "pending",
        }

        try:
            result = rag.run_rag(question, chunks, embeddings, model)
            evidence = []
            for rank, item in enumerate(result["hybrid_results"], 1):
                c = item["chunk"]
                evidence.append({
                    "rank": rank,
                    "company": c["company"],
                    "year": c["year"],
                    "section": c["section"],
                    "pdf_page": c["page"],
                    "chunk_id": c["id"],
                    "type": c["type"],
                    "rrf_score": item["rrf_score"],
                })

            retrieved_ids = {x["chunk_id"] for x in evidence}
            hits = [x for x in gold_ids if x in retrieved_ids]

            record.update({
                "status": "completed",
                "candidate_count": result["candidate_count"],
                "company_count": len(set(x["company"] for x in evidence)),
                "retrieval_reference_hits": hits,
                "retrieval_reference_total": len(gold_ids),
                "retrieval_reference_hit_rate": round(len(hits) / len(gold_ids), 4),
                "evidence": evidence,
                "answer": result["answer"],
                "answer_correctness": "待人工核验",
            })
            print(
                f"{qid} 完成：参考Chunk命中 "
                f"{len(hits)}/{len(gold_ids)}，"
                f"证据 {len(evidence)} 条",
                flush=True,
            )
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
            print(f"{qid} 出错：{exc}", flush=True)

        records.append(record)
        output.write_text(
            json.dumps({
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "method": "Vector + BM25 + RRF; per-company retrieval for multi-company queries",
                "note": "参考Chunk命中率不是语义召回率；答案正确性须单独核验。",
                "results": records,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n测试结束。结果已保存到：", output)
    print("成功题数：", sum(r["status"] == "completed" for r in records))
    print("出错题数：", sum(r["status"] == "error" for r in records))

if __name__ == "__main__":
    main()
