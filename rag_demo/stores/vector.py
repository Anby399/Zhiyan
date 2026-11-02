"""Milvus 连接及 dense / BM25 双向量 collection。"""

import os
from functools import lru_cache
from rag_demo.config import COLLECTION, MODELS

@lru_cache
def milvus():
    from pymilvus import MilvusClient

    return MilvusClient(uri=os.getenv("MILVUS_URI", "http://127.0.0.1:19530"))


def create_collection():
    from pymilvus import DataType, Function, FunctionType

    client = milvus()
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("text", DataType.VARCHAR, max_length=8192,
                     enable_analyzer=True, analyzer_params={"tokenizer": "jieba"})
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=MODELS["embedding_dimension"])
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="bm25", input_field_names=["text"],
                                 output_field_names=["sparse"],
                                 function_type=FunctionType.BM25))
    indexes = client.prepare_index_params()
    indexes.add_index("dense", index_type="AUTOINDEX", metric_type="COSINE")
    indexes.add_index("sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    client.create_collection(COLLECTION, schema=schema, index_params=indexes,
                             consistency_level="Strong")

