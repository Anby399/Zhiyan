"""按页读取 PDF 并切分，保留来源和页码。"""

from pathlib import Path
from rag_demo.config import PDF

def read_pdf(folder):
    """输入 PDF 目录，返回带 id、来源、页码和原文的片段列表。"""
    import pymupdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # 1. 优先按段落与中文标点切分；保留重叠内容，减少上下文在边界处丢失。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF["chunk_size"], chunk_overlap=PDF["chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    rows = []
    # 2. 固定文件遍历顺序；同一次全量导入中，片段 ID 按顺序分配。
    for path in sorted(Path(folder).rglob("*.pdf")):
        with pymupdf.open(path) as pdf:
            # 3. 页码从 1 开始，便于生成回答后回到原 PDF 核对引用。
            for page_num, page in enumerate(pdf, 1):
                for text in splitter.split_text(page.get_text()):
                    # 4. MySQL 与 Milvus 使用相同 ID，检索后据此回表查原文。
                    rows.append(dict(id=str(len(rows) + 1),
                                     source=str(path.relative_to(folder)),
                                     page=page_num, text=text))
    # 5. 提取结果为空时先中止，避免清除现有演示数据后才发现资料无效。
    if not rows:
        raise ValueError("未提取到文本：请放入文字版 PDF；扫描件需另做 OCR。")
    return rows

