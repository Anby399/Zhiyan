"""BGE-M3 编码与 BGE reranker 模型加载。"""

from functools import lru_cache

from rag_demo.config import DEVICE, EMBED_MODEL, MODELS, RERANK_MODEL


@lru_cache
def embedder():
    from sentence_transformers import SentenceTransformer

    # BGE-M3 的 dense 向量；BM25 稀疏向量由 Milvus 生成。
    return SentenceTransformer(EMBED_MODEL, device=DEVICE)


def embed(texts):
    return (
        embedder()
        .encode(texts, normalize_embeddings=True, batch_size=MODELS["batch_size"])
        .tolist()
    )


@lru_cache
def reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        RERANK_MODEL, device=DEVICE, max_length=MODELS["reranker_max_length"]
    )
