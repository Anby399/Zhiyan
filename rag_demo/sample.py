"""生成可用于课堂演示的五页 PDF。"""

from rag_demo.config import ROOT

def sample():
    import pymupdf

    # 1. 示例仅写入项目 data 目录，不依赖外部资料。
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    with pymupdf.open() as pdf:
        text = ("RAG先检索知识库，再把相关资料交给语言模型生成回答。\n"
                "MySQL保存文本片段和来源页码，Redis缓存相同问题的回答一小时。\n"
                "BGE-M3生成1024维稠密向量；Milvus保存向量并提供BM25关键词检索。\n"
                "WeightedRanker用0.6的向量权重和0.4的BM25权重融合两路召回结果。\n"
                "BGE reranker对问题和片段精排，选出前三条，ChatOpenAI依据它们回答。")
        # 2. 每个知识点单独成页，方便展示不同候选片段及其引用页码。
        for section in text.splitlines():
            page = pdf.new_page()
            # china-s 是内置中文字体，避免中文输出为空或乱码。
            page.insert_text((50, 70), "RAG教学资料", fontname="china-s", fontsize=18)
            page.insert_textbox((50, 110, 545, 750), section, fontname="china-s", fontsize=12)
        # 3. 保存后用 sample 命令输出路径，再用 ingest 读取同一份文件。
        pdf.save(folder / "demo.pdf")
    print(folder / "demo.pdf")

