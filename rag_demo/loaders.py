"""按页读取 PDF 并切分，保留来源和页码。"""

from pathlib import Path
from rag_demo.config import PDF

def read_pdf(folder):
    import pymupdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PDF["chunk_size"], chunk_overlap=PDF["chunk_overlap"],
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    rows = []
    for path in sorted(Path(folder).rglob("*.pdf")):
        with pymupdf.open(path) as pdf:
            for page_num, page in enumerate(pdf, 1):
                for text in splitter.split_text(page.get_text()):
                    rows.append(dict(id=str(len(rows) + 1),
                                     source=str(path.relative_to(folder)),
                                     page=page_num, text=text))
    if not rows:
        raise ValueError("未提取到文本：请放入文字版 PDF；扫描件需另做 OCR。")
    return rows

