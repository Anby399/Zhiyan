"""双路召回 → WeightedRanker 融合 → MySQL 回表 → BGE 精排。"""

from rag_demo.config import COLLECTION, RETRIEVAL
from rag_demo.models import embed, reranker
from rag_demo.stores.mysql import mysql
from rag_demo.stores.vector import milvus


def retrieve(question):
    from pymilvus import AnnSearchRequest, WeightedRanker

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
    hits = milvus().hybrid_search(
        COLLECTION,
        reqs=requests,
        ranker=WeightedRanker(RETRIEVAL["dense_weight"], RETRIEVAL["bm25_weight"]),
        limit=RETRIEVAL["fusion_k"],
    )[0]
    if not hits:
        return []
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
    scores = reranker().predict([(question, row["text"]) for row in rows])
    for row, score in zip(rows, scores):
        row["score"] = float(score)
    return sorted(rows, key=lambda row: row["score"], reverse=True)[
        : RETRIEVAL["top_k"]
    ]
