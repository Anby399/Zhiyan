"""双路召回 → WeightedRanker 融合 → MySQL 回表 → BGE 精排。"""

from rag_demo.config import COLLECTION, RETRIEVAL
from rag_demo.models import embed, reranker
from rag_demo.stores.mysql import mysql
from rag_demo.stores.vector import milvus


def retrieve(question):
    """输入问题，返回 BGE 精排后的 top_k 条原文片段。"""
    from pymilvus import AnnSearchRequest, WeightedRanker

    # 1. 稠密检索理解语义；BM25 接收原问题，使用服务端分词匹配关键词。
    requests = [
        AnnSearchRequest(
            embed([question]),
            "dense",
            {"metric_type": "COSINE"},
            limit=RETRIEVAL["recall_k"],
        ),
        AnnSearchRequest(
            [question], "sparse", {"metric_type": "BM25"}, limit=RETRIEVAL["recall_k"]
        ),
    ]
    # 2. 归一化两路分数并加权融合；[0] 取出本次唯一问题的命中列表。
    hits = milvus().hybrid_search(
        COLLECTION,
        reqs=requests,
        ranker=WeightedRanker(RETRIEVAL["dense_weight"], RETRIEVAL["bm25_weight"]),
        limit=RETRIEVAL["fusion_k"],
    )[0]
    if not hits:
        return []
    # 3. 命中只需要 ID；SQL 使用参数占位符传值，回查文本和来源。
    ids = [str(hit["id"]) for hit in hits]
    with mysql().cursor() as cur:
        cur.execute(
            "SELECT * FROM teaching_rag_chunks WHERE id IN ("
            + ",".join(["%s"] * len(ids))
            + ")",
            ids,
        )
        rows = cur.fetchall()
    # BGE reranker：直接给“问题、片段”打分，与 WeightedRanker 的融合分不同。
    if not rows:
        return []
    # 4. 一条候选组成一个“问题、原文”对，分数与候选列表按位置对应。
    scores = reranker().predict([(question, row["text"]) for row in rows])
    for row, score in zip(rows, scores):
        row["score"] = float(score)
    # 5. SQL 的 IN 查询不保证顺序；最终以精排模型的分数降序取 top_k。
    return sorted(rows, key=lambda row: row["score"], reverse=True)[
        : RETRIEVAL["top_k"]
    ]
