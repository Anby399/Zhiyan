"""配置类型、边界、字段关系与环境变量优先级。"""

import unittest

from pydantic import ValidationError

from rag_demo.config import Config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.data = Config.load(environ={}).model_dump()

    def test_load_typed_sections(self):
        config = Config.model_validate(self.data)
        self.assertEqual(config.pdf.chunk_size, 500)
        self.assertEqual(config.models.embedding_dimension, 1024)

    def test_invalid_types_ranges_and_unknown_fields(self):
        cases = [
            ("pdf", "chunk_size", "500"),
            ("pdf", "chunk_size", True),
            ("pdf", "chunk_size", 2001),
            ("models", "embedding", "   "),
            ("models", "embedding_dimension", 768),
            ("retrieval", "dense_weight", 1.1),
            ("retrieval", "bm25_weight", float("nan")),
            ("storage", "cache_ttl", 0),
            ("llm", "timeout", -1),
            ("llm", "max_retries", -1),
            ("pdf", "chunk_szie", 500),
        ]
        for section, field, value in cases:
            with self.subTest(section=section, field=field):
                data = Config.model_validate(self.data).model_dump()
                data[section][field] = value
                with self.assertRaises(ValidationError):
                    Config.model_validate(data)

    def test_cross_field_constraints(self):
        for section, changes in [
            ("pdf", {"chunk_overlap": 500}),
            ("retrieval", {"dense_weight": 0.0, "bm25_weight": 0.0}),
            ("retrieval", {"top_k": 11}),
        ]:
            with self.subTest(changes=changes):
                data = Config.model_validate(self.data).model_dump()
                data[section].update(changes)
                with self.assertRaises(ValidationError):
                    Config.model_validate(data)

    def test_environment_overrides_are_validated(self):
        config = Config.load(environ={"EMBED_MODEL": "local/bge-m3", "DEVICE": "cuda:0"})
        self.assertEqual(config.models.embedding, "local/bge-m3")
        self.assertEqual(config.models.device, "cuda:0")
        with self.assertRaises(ValidationError):
            Config.load(environ={"DEVICE": "invalid-device"})


if __name__ == "__main__":
    unittest.main()
