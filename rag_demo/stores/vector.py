"""Milvus 连接及 dense / BM25 双向量 collection。"""

import os
from functools import lru_cache
from rag_demo.config import COLLECTION, MODELS

@lru_cache
def milvus():
    from pymilvus import MilvusClient

    # 1. 使用独立 Milvus 服务；本例的原生 BM25 流程不使用本地 Lite 文件。
    return MilvusClient(uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))


def create_collection():
    from pymilvus import DataType, Function, FunctionType

    client = milvus()
    # 2. 自定义主键，与 MySQL 的片段 ID 一致；关闭动态字段，便于讲解 schema。
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    # 3. text 启用 jieba 分词；dense 由 BGE-M3 生成，sparse 由 BM25 自动生成。
    schema.add_field("text", DataType.VARCHAR, max_length=8192,
                     enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=MODELS["embedding_dimension"])
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    # 4. 声明 text → sparse 的 BM25 函数，写入和查询时由服务端执行。
    schema.add_function(Function(name="bm25", input_field_names=["text"],
                                 output_field_names=["sparse"],
                                 function_type=FunctionType.BM25))
    # 5. 两个向量字段分别建立索引，搜索度量需与 retrieval.py 中的请求一致。
    indexes = client.prepare_index_params()
    indexes.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
    indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    # 6. 强一致性便于教学：后续查询能看到此前已经完成的写入。
    client.create_collection(COLLECTION, schema=schema, index_params=indexes,
                             consistency_level="Strong")

