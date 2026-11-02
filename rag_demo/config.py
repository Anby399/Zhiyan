"""读取配置并在启动时使用 Pydantic 校验类型、取值与字段关系。"""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator

# 1. 用配置文件所在位置定位项目，避免从不同目录启动时找不到 .env。
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class ConfigSection(BaseModel):
    """所有配置段共享规则：拒绝隐式类型转换、未知字段，并去除字符串首尾空格。"""

    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)


class PDFConfig(ConfigSection):
    # 字符数上限让中文片段编码后仍能放入 Milvus 的文本字段。
    chunk_size: int = Field(gt=0, le=2000)
    chunk_overlap: int = Field(ge=0)

    # 单字段通过类型校验后，再检查两个字段之间的关系。
    @model_validator(mode="after")
    def check_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size。")
        return self


class ModelsConfig(ConfigSection):
    embedding: str = Field(min_length=1)
    reranker: str = Field(min_length=1)
    device: str = Field(pattern=r"^(cpu|mps|cuda(:[0-9]+)?)$")
    # 当前示例固定使用 BGE-M3；维度必须与 Milvus 的向量 schema 一致。
    embedding_dimension: Literal[1024]
    batch_size: int = Field(gt=0)
    reranker_max_length: int = Field(gt=0)


class RetrievalConfig(ConfigSection):
    recall_k: int = Field(gt=0)
    fusion_k: int = Field(gt=0)
    top_k: int = Field(gt=0)
    dense_weight: float = Field(ge=0, le=1, allow_inf_nan=False)
    bm25_weight: float = Field(ge=0, le=1, allow_inf_nan=False)

    # 每个权重可独立调整；不要求和为 1，但不能全部为 0。
    @model_validator(mode="after")
    def check_ranking(self) -> Self:
        if self.dense_weight + self.bm25_weight == 0:
            raise ValueError("两路融合权重不能同时为 0。")
        if self.top_k > self.fusion_k:
            raise ValueError("top_k 不能大于 fusion_k。")
        return self


class StorageConfig(ConfigSection):
    # collection 名使用 Milvus 支持的标识符格式。
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
        # 2. TOML 按节读取为嵌套字典，此时还未创建类型化对象。
        with Path(path).open("rb") as file:
            data = tomllib.load(file)
        # 3. 环境变量覆盖模型配置；传入 environ 便于测试而不修改真实环境。
        env = os.environ if environ is None else environ
        for name, field in (("EMBED_MODEL", "embedding"),
                            ("RERANK_MODEL", "reranker"), ("DEVICE", "device")):
            if env.get(name):
                data.setdefault("models", {})[field] = env[name]
        # 4. 一次校验所有嵌套字段，失败时 ValidationError 会给出字段路径。
        return cls.model_validate(data)


# 5. 启动时立即校验，确保错误配置不会进入模型加载或数据导入阶段。
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
