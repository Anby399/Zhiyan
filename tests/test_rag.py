"""离线验证重构后的模块协作，不下载模型或连接数据库。"""

import io
import copy
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from pymilvus import MilvusClient

from rag_demo import pipeline, retrieval
from rag_demo.config import ROOT, COLLECTION, PREFIX, STORAGE
from rag_demo.loaders import read_pdf
from rag_demo.stores import vector


class RagTests(unittest.TestCase):
    def test_sample_pdf_keeps_source_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            shutil.copyfile(ROOT / "data" / "demo.pdf", folder + "/demo.pdf")
            rows = read_pdf(folder)
        self.assertEqual({r["page"] for r in rows}, set(range(1, 6)))
        self.assertTrue(all(r["source"] == "demo.pdf" for r in rows))

    def test_schema_contains_bm25_and_dense_vectors(self):
        client = Mock()
        client.create_schema = MilvusClient.create_schema
        client.prepare_index_params = MilvusClient.prepare_index_params
        with patch.object(vector, "milvus", return_value=client):
            vector.create_collection()
        schema = client.create_collection.call_args.kwargs["schema"].to_dict()
        self.assertEqual({f["name"] for f in schema["fields"]},
                         {"id", "text", "dense", "sparse"})
        self.assertEqual(schema["functions"][0]["input_field_names"], ["text"])

    def test_cache_hit_skips_retrieval(self):
        cache = Mock()
        cache.get.side_effect = ["version-1", "cached-answer"]
        with patch.object(pipeline, "redis_client", return_value=cache), \
                patch.object(pipeline, "retrieve") as retrieve, \
                redirect_stdout(io.StringIO()) as output:
            pipeline.ask("question")
        retrieve.assert_not_called()
        self.assertIn("cached-answer", output.getvalue())

    def test_changed_retrieval_config_uses_a_new_cache_key(self):
        cache = Mock()
        cache.get.side_effect = ["version-1", "cached", "version-1", "cached"]
        changed = copy.deepcopy(pipeline.SETTINGS)
        changed["retrieval"]["dense_weight"] = 0.9
        with patch.object(pipeline, "redis_client", return_value=cache), \
                redirect_stdout(io.StringIO()):
            pipeline.ask("same-question")
            with patch.object(pipeline, "SETTINGS", changed):
                pipeline.ask("same-question")
        self.assertNotEqual(cache.get.call_args_list[1], cache.get.call_args_list[3])

    def test_failed_ingest_never_marks_ready(self):
        db, cache, client = Mock(), Mock(), Mock()
        db.cursor.return_value.__enter__ = Mock(return_value=Mock())
        db.cursor.return_value.__exit__ = Mock(return_value=False)
        rows = [dict(id="1", source="demo.pdf", page=1, text="example")]
        with patch.object(pipeline, "read_pdf", return_value=rows), \
                patch.object(pipeline, "mysql", return_value=db), \
                patch.object(pipeline, "redis_client", return_value=cache), \
                patch.object(pipeline, "milvus", return_value=client), \
                patch.object(pipeline, "create_collection"), \
                patch.object(pipeline, "embed", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                pipeline.ingest("unused")
        cache.delete.assert_called_once_with(PREFIX + "ready")
        cache.set.assert_not_called()

    def test_reranker_orders_mysql_results(self):
        client, db, cursor, model = Mock(), Mock(), Mock(), Mock()
        client.hybrid_search.return_value = [[{"id": "1"}, {"id": "2"}]]
        db.cursor.return_value.__enter__ = Mock(return_value=cursor)
        db.cursor.return_value.__exit__ = Mock(return_value=False)
        cursor.fetchall.return_value = [dict(id="1", text="low"), dict(id="2", text="high")]
        model.predict.return_value = [0.1, 0.9]
        with patch.object(retrieval, "milvus", return_value=client), \
                patch.object(retrieval, "mysql", return_value=db), \
                patch.object(retrieval, "embed", return_value=[[0.0] * 1024]), \
                patch.object(retrieval, "reranker", return_value=model):
            rows = retrieval.retrieve("question")
        self.assertEqual([r["id"] for r in rows], ["2", "1"])
        self.assertEqual(client.hybrid_search.call_args.args[0], COLLECTION)

    def test_answer_includes_citations_and_is_cached(self):
        cache, model = Mock(), Mock()
        cache.get.side_effect = ["version-1", None]
        model.invoke.return_value.content = "answer [1]"
        rows = [dict(source="demo.pdf", page=2, text="reference")]
        with patch.object(pipeline, "redis_client", return_value=cache), \
                patch.object(pipeline, "retrieve", return_value=rows), \
                patch("langchain_openai.ChatOpenAI", return_value=model), \
                redirect_stdout(io.StringIO()) as output:
            pipeline.ask("question")
        self.assertIn("answer [1]", output.getvalue())
        self.assertIn("demo.pdf", output.getvalue())
        self.assertEqual(cache.setex.call_args.args[1], STORAGE["cache_ttl"])


if __name__ == "__main__":
    unittest.main()
