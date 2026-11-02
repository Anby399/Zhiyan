"""命令参数解析。"""

import argparse
from rag_demo.config import ROOT
from rag_demo.pipeline import ingest, ask
from rag_demo.sample import sample


def main():
    # 1. 三个子命令分别演示资料生成、导入和问答。
    parser = argparse.ArgumentParser(description="最小 RAG 教学演示")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("sample", help="生成教学 PDF")
    commands.add_parser("ingest", help="全量重建本示例的索引").add_argument(
        "folder", nargs="?", default=str(ROOT / "data"))
    commands.add_parser("ask", help="检索并问答").add_argument("question")
    # 2. 解析参数后分派到流程函数，入口本身不承担检索或数据库逻辑。
    args = parser.parse_args()
    if args.command == "sample":
        sample()
    elif args.command == "ingest":
        ingest(args.folder)
    else:
        ask(args.question)
