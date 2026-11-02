"""生成可用于课堂演示的五页 PDF。"""

from rag_demo.config import ROOT

def sample():
    import pymupdf

    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    with pymupdf.open() as pdf:
        text = ("RAG先检索知识库，再把相关资料交给语言模型生成回答。\n"
                "MySQL保存文本片段和来源页码，Redis缓存相同问题的回答一小时。\n"
                "BGE-M3生成1024维稠密向量；Milvus保存向量并提供BM25关键词检索。\n"
                "WeightedRanker用0.6的向量权重和0.4的BM25权重融合两路召回结果。\n"
                "BGE reranker对问题和片段精排，选出前三条，ChatOpenAI依据它们回答。")
        for section in text.splitlines():
            page = pdf.new_page()
            page.insert_text((50, 70), "RAG教学资料", fontname="china-s", fontsize=18)
            page.insert_textbox((50, 110, 545, 750), section, fontname="china-s", fontsize=12)
        pdf.save(folder / "demo.pdf")
    print(folder / "demo.pdf")

