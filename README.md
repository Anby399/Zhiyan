# 最小 RAG 教学演示

按配置、解析、模型、存储、检索和流程编排拆分，保持各模块简短。

教学文档：[四页项目流程图（PDF）](docs/项目流程图.pdf) · [整个项目的文字流程（Markdown）](docs/项目文字流程.md)。

## 项目结构

```text
Zhiyan/
├── main.py                 # 推荐的命令入口
├── rag.py                  # 兼容旧命令的入口
├── config.toml             # 分块、模型、检索、缓存、LLM 参数
├── .env.example            # 连接信息和密钥模板
├── .env                    # 本地配置，不提交
├── compose.yaml            # MySQL、Redis、Milvus 及其依赖
├── requirements.txt
├── rag_demo/
│   ├── __init__.py
│   ├── __main__.py         # 支持 python -m rag_demo
│   ├── cli.py              # 参数解析
│   ├── config.py           # 统一加载配置与参数校验
│   ├── loaders.py          # PDF 解析与分块
│   ├── models.py           # BGE-M3 / BGE reranker 加载
│   ├── retrieval.py        # 双路召回、融合、精排
│   ├── pipeline.py         # 导入与问答流程编排
│   ├── sample.py           # 示例资料生成
│   └── stores/
│       ├── __init__.py
│       ├── mysql.py        # MySQL 连接与建表
│       ├── cache.py        # Redis 连接
│       └── vector.py       # Milvus 连接、schema 与索引
├── data/
│   └── demo.pdf            # 五页教学资料
└── tests/
    └── test_rag.py          # 不连接外部服务的协作测试
```

## 配置说明

- `rag_demo/config.py` 中的 Pydantic `Config` 类提供嵌套类型配置：`from rag_demo.config import config` 后访问 `config.pdf.chunk_size`。启动时校验字段类型、数值范围、未知字段及分块重叠、检索条数和权重关系；错误会抛出带字段位置的 `ValidationError`。整数参数不接受字符串或布尔值。
- `config.toml`：非敏感参数。默认按 500 字符切分、重叠 80 字符；两路各召回 10 条、融合 10 条、精排取 3 条；权重 0.6/0.4，缓存 3600 秒。
- `.env`：MySQL、Redis、Milvus 连接信息，以及 OpenAI 密钥、地址与模型名。现有文件保留不覆盖。
- `EMBED_MODEL`、`RERANK_MODEL`、`DEVICE` 如在环境变量或 `.env` 中设置，优先于 `config.toml` 的模型配置；已有系统环境变量优先于 `.env`。
- 修改配置后重新执行命令生效。问答缓存键包含 `config.toml` 参数，调整检索参数后自动切换缓存；修改分块或编码器后仍需重新导入。

建议课堂阅读顺序：`cli.py → pipeline.py → loaders.py / retrieval.py → models.py / stores/`。

各模块已按实际处理顺序添加中文教学注释，说明输入、处理目的和输出。项目协作约定见 `AGENTS.md`：后续代码修改同时维护注释，使用 `lang_env` 验证后提交并推送到现有 GitHub 远程分支。

```text
导入：PDF → PyMuPDF → 分块 → MySQL（片段、文件名、页码）
                          → BGE-M3 → Milvus（dense + 内置 BM25 sparse）
问答：Redis 缓存 → 未命中 → dense / BM25 各召回 10 条
     → WeightedRanker(0.6, 0.4) 融合取 10 条
     → BGE reranker 精排取 3 条 → ChatOpenAI → 带引用回答 → Redis 缓存 1 小时
```

| 组件 | 本例职责 |
| --- | --- |
| MySQL | 保存片段原文和来源，检索命中后按 ID 回表 |
| Redis | 缓存问答结果；导入版本隔离旧缓存 |
| Milvus | 保存 BGE-M3 的 1024 维向量，执行原生 BM25 与混合检索 |
| BM25 | Milvus 内置函数自动计算稀疏向量，使用 jieba 分词 |
| PyMuPDF | 按页提取 PDF 文本，保留页码 |
| BGE-M3 | 本地 `BAAI/bge-m3`，通过 SentenceTransformer 生成 dense 向量 |
| WeightedRanker | 用户所述 weightranker，归一化并融合两路检索分数 |
| BGE reranker | 本地 `BAAI/bge-reranker-v2-m3`，通过 CrossEncoder 逐对打分 |
| ChatOpenAI | 正确导入为 `from langchain_openai import ChatOpenAI` |

BGE-M3 自身的 learned sparse 不是 BM25。本例只使用其 dense 输出，BM25 由 Milvus 负责。
WeightedRanker 是召回分数融合，BGE reranker 是模型精排，代码中保留了两个独立阶段。

## 1. 使用现有 lang_env

在 PowerShell 中运行：

```powershell
cd F:\Zhiyan
conda activate lang_env
python -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

将 `OPENAI_API_KEY` 改成自己的密钥，按需修改 `OPENAI_MODEL`、`OPENAI_BASE_URL`。
首次创建 `.env` 时复制一次即可，已有配置不要覆盖。
若 PowerShell 未初始化 conda，可直接使用指定解释器：

```powershell
& F:\heima\anaconda3\envs\lang_env\python.exe -m pip install -r requirements.txt
& F:\heima\anaconda3\envs\lang_env\python.exe rag.py --help
```

Python 需 3.11+（使用标准库 tomllib）；本机 `lang_env` 为 3.12。客户端约束为 PyMilvus 2.6.x，与 Compose 的 Milvus 2.6.x 对齐，安装会调整现有不兼容的 PyMilvus 版本。其余依赖使用版本范围，未锁定全部间接依赖。

## 2. 启动服务

先打开 Docker Desktop，确认使用 Linux containers，再执行：

```powershell
docker compose up -d --wait --wait-timeout 240
docker compose ps
```

Compose 包含 MySQL、Redis、Milvus，以及 Milvus 所需的 etcd、MinIO。
数据使用 Docker 命名卷持久化。主机端口为 3306、6379、19530；已有服务占用时修改 Compose 端口及 `.env`。
数据库账号仅供本地教学，服务端口绑定到 `127.0.0.1`。

## 3. 生成资料、导入、问答

```powershell
python main.py sample
python main.py ingest
python main.py ask "WeightedRanker 和 BGE reranker 有什么区别？"
python main.py ask "WeightedRanker 和 BGE reranker 有什么区别？"
```

第二次相同提问会显示 `[Redis 缓存命中]`，不再加载模型或调用 LLM。
输出包含答案、引用编号和原文片段，便于对照讲解。
也可以把自己的文字版 PDF 放入 `data/`，或执行 `python main.py ingest "D:\课程资料"`。

首次运行会下载两个模型，需要网络和数 GB 磁盘空间。默认 CPU，速度较慢；已有 CUDA 版 PyTorch 时可设 `DEVICE=cuda`。
支持将 `EMBED_MODEL`、`RERANK_MODEL` 改为本地模型目录；编码器必须是 1024 维 BGE-M3。
更换编码器后重新导入。Redis 缓存过期一小时；修改提示词后可重新导入以切换缓存版本；修改配置中的融合权重会自动切换缓存。

停止服务但保留数据：

```powershell
docker compose down
```

## 教学边界

- `ingest` 是**全量重建**：替换 `teaching_rag_chunks` 表的数据及同名 Milvus collection，只操作本例资源。请使用独立的演示数据库。
- 按单人、串行命令设计，不要同时执行导入与问答。导入失败会保持未就绪状态；修复后重新执行导入。
- Redis 未启用持久化，重启后需要重新导入以建立就绪版本；不会复用可能过期的答案。
- 扫描 PDF 需要另加 OCR；表格与复杂版面暂按提取出的纯文本处理。
- 未实现事务化跨库发布、增量更新、会话历史、权限控制、生产监控，以保持教学流程清晰。

## 官方接口参考

- [Milvus 原生 BM25](https://milvus.io/docs/v2.5.x/full-text-search.md)
- [Milvus 混合检索](https://milvus.io/docs/v2.5.x/multi-vector-search.md)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [BGE reranker 及 CrossEncoder 用法](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [LangChain ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai)

本示例采用服务器原生 BM25，不使用 Milvus Lite。

## 本机验证记录

已在 `F:\heima\anaconda3\envs\lang_env` 安装本项目依赖，全部组件及模型加载类可导入。模块拆分后可执行以下离线测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖 PDF 页码、Milvus schema、缓存绕过检索、失败导入不发布就绪状态、精排顺序和带引用回答缓存。外部服务及模型输出使用模拟对象，不代表端到端验证。
Docker 引擎未启动，`.env` 保留待填写的密钥；未下载 BGE 模型或执行真实端到端问答。
环境整体 `pip check` 另报告已有的 botocore/python-dateutil、dataclasses-json/marshmallow、langchain-mcp-adapters/mcp 版本冲突，未为本项目调整这些无关依赖。
