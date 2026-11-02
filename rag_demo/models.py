"""BGE-M3 编码与 BGE reranker 模型加载。"""

from functools import lru_cache

from rag_demo.config import DEVICE, EMBED_MODEL, MODELS, RERANK_MODEL


# 延迟加载且复用模型：命中 Redis 缓存时无需加载模型权重。
@lru_cache
def embedder():
    from sentence_transformers import SentenceTransformer

    # BGE-M3 的 dense 向量；BM25 稀疏向量由 Milvus 生成。
    return SentenceTransformer(EMBED_MODEL, device=DEVICE)


def embed(texts):
    """文本列表 → 归一化稠密向量列表，供 Milvus 写入或查询。"""
    # encode 批量推理；tolist 将 NumPy 数组转为客户端可接收的 Python 列表。
    return (
        embedder()
        .encode(texts, normalize_embeddings=True, batch_size=MODELS["batch_size"])
        .tolist()
    )


# CrossEncoder 同时读取问题和片段，输出相关性分数，不生成检索向量。
@lru_cache
def reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        RERANK_MODEL, device=DEVICE, max_length=MODELS["reranker_max_length"]
    )
