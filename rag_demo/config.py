"""读取配置并在启动时使用 Pydantic 校验类型、取值与字段关系。"""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class ConfigSection(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)


class PDFConfig(ConfigSection):
    chunk_size: int = Field(gt=0, le=2000)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size。")
        return self


class ModelsConfig(ConfigSection):
    embedding: str = Field(min_length=1)
    reranker: str = Field(min_length=1)
    device: str = Field(pattern=r"^(cpu|mps|cuda(:[0-9]+)?)$")
    embedding_dimension: Literal[1024]
    batch_size: int = Field(gt=0)
    reranker_max_length: int = Field(gt=0)


class RetrievalConfig(ConfigSection):
    recall_k: int = Field(gt=0)
    fusion_k: int = Field(gt=0)
    top_k: int = Field(gt=0)
    dense_weight: float = Field(ge=0, le=1, allow_inf_nan=False)
    bm25_weight: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def check_ranking(self) -> Self:
        if self.dense_weight + self.bm25_weight == 0:
            raise ValueError("两路融合权重不能同时为 0。")
        if self.top_k > self.fusion_k:
            raise ValueError("top_k 不能大于 fusion_k。")
        return self


class StorageConfig(ConfigSection):
    collection: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
    cache_prefix: str = Field(min_length=1)
    cache_ttl: int = Field(gt=0)
    ingest_batch_size: int = Field(gt=0)


class LLMConfig(ConfigSection):
    temperature: float = Field(ge=0, le=2, allow_inf_nan=False)
    timeout: float = Field(gt=0, allow_inf_nan=False)
    max_retries: int = Field(ge=0)


class Config(ConfigSection):
    """对应 config.toml 的嵌套配置，可用 config.pdf.chunk_size 访问。"""

    pdf: PDFConfig
    models: ModelsConfig
    retrieval: RetrievalConfig
    storage: StorageConfig
    llm: LLMConfig

    @classmethod
    def load(cls, path: Path = ROOT / "config.toml",
             environ: Mapping[str, str] | None = None) -> Self:
        """先加载 TOML，再应用模型环境变量，最终统一校验。"""
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
        env = os.environ if environ is None else environ
        for name, field in (("EMBED_MODEL", "embedding"),
                            ("RERANK_MODEL", "reranker"), ("DEVICE", "device")):
            if env.get(name):
                data.setdefault("models", {})[field] = env[name]
        return cls.model_validate(data)


config = Config.load()

# 兼容现有模块的字典接口，值均来自已通过校验的 Config 实例。
SETTINGS = config.model_dump()
PDF = SETTINGS["pdf"]
MODELS = SETTINGS["models"]
RETRIEVAL = SETTINGS["retrieval"]
STORAGE = SETTINGS["storage"]
LLM = SETTINGS["llm"]
COLLECTION = config.storage.collection
PREFIX = config.storage.cache_prefix
EMBED_MODEL = config.models.embedding
RERANK_MODEL = config.models.reranker
DEVICE = config.models.device
