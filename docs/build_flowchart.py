"""生成四页教学流程图；运行：python docs/build_flowchart.py。"""

import math
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A3, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


# 1. 使用矢量绘图和嵌入式中文字体，放大或打印时保持文字清晰。
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "项目流程图.pdf"
W, H = landscape(A3)
INK = HexColor("#18334D")
LINE = HexColor("#52718A")
BLUE = HexColor("#EAF3FC")
GREEN = HexColor("#E7F5F0")
AMBER = HexColor("#FFF2D9")
RED = HexColor("#FCECEC")


def register_font():
    """优先使用 Windows 黑体；其他系统可提供 docs/SimHei.ttf。"""
    for path in (HERE / "SimHei.ttf", Path("C:/Windows/Fonts/simhei.ttf")):
        if path.exists():
            pdfmetrics.registerFont(TTFont("TeachingCN", str(path)))
            return
    raise FileNotFoundError("请安装中文字体，或将 SimHei.ttf 放入 docs 目录。")


def text(x, y, value, size=14, color=INK, center=False):
    """以下坐标统一从左上角开始，绘制时转换为 PDF 的左下角坐标。"""
    c.setFillColor(color)
    c.setFont("TeachingCN", size)
    (c.drawCentredString if center else c.drawString)(x, H - y, value)


def box(x, y, title, lines=(), fill=BLUE, w=280, h=90):
    """每个节点标题说明处理动作，第二层文字给出代码模块或数据。"""
    c.setStrokeColor(LINE)
    c.setFillColor(fill)
    c.setLineWidth(1)
    c.roundRect(x, H - y - h, w, h, 10, fill=1, stroke=1)
    for value, size in [(title, 17)] + [(line, 13) for line in lines]:
        assert pdfmetrics.stringWidth(value, "TeachingCN", size) < w - 20, value
    assert 32 + len(lines) * 19 < h, title
    text(x + w / 2, y + 27, title, 17, center=True)
    for i, line in enumerate(lines):
        text(x + w / 2, y + 49 + i * 19, line, 13, center=True)


def decision(x, y, title, detail="", w=280, h=110):
    """菱形用于判断；是、否分别标在对应的出线上。"""
    p = c.beginPath()
    for i, (px, py) in enumerate(((x + w / 2, y), (x + w, y + h / 2),
                                 (x + w / 2, y + h), (x, y + h / 2))):
        (p.moveTo if i == 0 else p.lineTo)(px, H - py)
    p.close()
    c.setFillColor(AMBER)
    c.setStrokeColor(LINE)
    c.drawPath(p, fill=1, stroke=1)
    text(x + w / 2, y + h / 2 - 1, title, 16, center=True)
    if detail:
        text(x + w / 2, y + h / 2 + 19, detail, 11, center=True)


def arrow(points, label="", at=None):
    """按指定折线连接节点，在终点绘制箭头，避免线穿过节点文字。"""
    c.setStrokeColor(LINE)
    c.setLineWidth(1.5)
    p = c.beginPath()
    p.moveTo(points[0][0], H - points[0][1])
    for x, y in points[1:]:
        p.lineTo(x, H - y)
    c.drawPath(p)
    (x0, y0), (x, y) = points[-2:]
    angle = math.atan2(y - y0, x - x0)
    head = c.beginPath()
    head.moveTo(x, H - y)
    for turn in (0.45, -0.45):
        head.lineTo(x - 9 * math.cos(angle + turn), H - y + 9 * math.sin(angle + turn))
    head.close()
    c.setFillColor(LINE)
    c.drawPath(head, fill=1, stroke=0)
    if label:
        assert at is not None
        width = pdfmetrics.stringWidth(label, "TeachingCN", 12)
        c.setFillColor(white)
        c.rect(at[0] - 4, H - at[1] - 3, width + 8, 17, fill=1, stroke=0)
        text(*at, label, size=12)


def header(number, title, subtitle):
    """统一页眉、页脚和配色，四页可独立用于课堂讲解。"""
    c.setFillColor(HexColor("#F8FAFD"))
    c.rect(0, 0, W, H, fill=1, stroke=0)
    text(50, 48, "ZHIYAN  /  RAG TEACHING DEMO", 13, LINE)
    text(50, 91, title, 30)
    text(50, 119, subtitle, 13, LINE)
    c.setStrokeColor(HexColor("#CFDCE7"))
    c.line(50, H - 793, W - 50, H - 793)
    text(50, 818, "对应项目当前代码与默认配置；详细步骤见 docs/项目文字流程.md", 11, LINE)
    text(W - 55, 818, f"{number} / 4", 12, LINE, center=True)


def overview():
    header(1, "项目总览：配置、命令与基础服务", "先校验配置，再选择演示命令；数据导入与用户问答是两条独立流程。")
    box(50, 160, "配置输入", ("config.toml + .env / 环境变量", "参数、连接地址、模型配置"))
    box(455, 160, "Pydantic Config 校验", ("类型、范围与字段关系", "rag_demo/config.py"))
    box(860, 160, "CLI 命令分派", ("python main.py <command>", "rag_demo/cli.py"))
    arrow([(330, 205), (455, 205)])
    arrow([(735, 205), (860, 205)])
    for center in (190, 595, 1000):
        arrow([(1000, 250), (1000, 295), (center, 295), (center, 340)])
    box(50, 340, "sample：生成资料", ("PyMuPDF 生成五页教学 PDF", "rag_demo/sample.py"))
    box(455, 340, "ingest：全量导入", ("解析、分块、编码、双库存储", "详见第 2 页"))
    box(860, 340, "ask：带引用问答", ("缓存、混合检索、精排、生成", "详见第 3、4 页"))
    arrow([(190, 430), (190, 525)])
    arrow([(595, 430), (595, 525)])
    arrow([(1000, 430), (1000, 525)])
    box(50, 525, "data/demo.pdf", ("仅创建本地资料", "下一步执行 ingest"), GREEN)
    box(455, 525, "MySQL + Milvus", ("原文与来源 / dense + sparse", "Redis 发布导入版本"), GREEN)
    box(860, 525, "答案 + 引用原文", ("ChatOpenAI 依据上下文生成", "Redis 缓存默认 3600 秒"), GREEN)
    box(50, 680, "Docker Compose 基础服务", (
        "MySQL :3306  |  Redis :6379  |  Milvus :19530  |  etcd 与 MinIO 为 Milvus 提供基础依赖",
        "蓝色：处理步骤    绿色：结果或存储    黄色：条件判断    红色：提前结束；所有端口仅绑定本机",
    ), w=1090, h=90)
    c.showPage()


def ingestion():
    header(2, "资料导入：PDF 如何成为可检索的知识库", "按箭头从上排向右、中排向左、下排向右阅读；ingest 会替换本例已有数据。")
    box(50, 160, "1. 接收 PDF 目录", ("python main.py ingest [folder]", "默认目录：data"))
    box(455, 160, "2. 逐页提取并分块", ("PyMuPDF + 文本切分器", "500 字符 / 重叠 80 字符"))
    decision(860, 150, "提取到片段？", "保留 id、来源、页码、原文")
    arrow([(330, 205), (455, 205)])
    arrow([(735, 205), (860, 205)])
    box(860, 280, "无文本：报错并结束", ("尚未清除旧索引或就绪标记",), RED, h=65)
    arrow([(1000, 260), (1000, 280)], "否", (1012, 275))
    box(860, 385, "3. 准备本例存储", ("连接服务、删除 ready", "重建 Milvus；清空 MySQL"))
    arrow([(1140, 205), (1165, 205), (1165, 430), (1140, 430)], "是", (1147, 320))
    box(455, 385, "4. BGE-M3 批量编码", ("每批导入 16 个片段", "生成 1024 维归一化 dense"))
    box(50, 385, "5. 先写 MySQL", ("id / source / page / text", "保存原文及可追溯来源"), GREEN)
    arrow([(860, 430), (735, 430)])
    arrow([(455, 430), (330, 430)])
    box(50, 590, "6. 再写 Milvus", ("id / text / dense", "BM25 根据 text 自动生成 sparse"), GREEN)
    arrow([(190, 475), (190, 590)])
    box(455, 590, "7. 完成全部批次", ("重复步骤 4-6 直到资料写完", "执行 Milvus flush"))
    box(860, 590, "8. 发布就绪版本", ("Redis ready = 新 UUID", "新版本隔离旧答案缓存"), GREEN)
    arrow([(330, 635), (455, 635)])
    arrow([(735, 635), (860, 635)])
    text(55, 730, "教学要点：两库使用同一片段 ID；BM25 sparse 由 Milvus 生成，BGE-M3 仅提供 dense。", 15)
    text(55, 757, "失败处理：删除 ready 后若导入失败，不发布新版本；修复后重新导入。两库写入没有跨库事务。", 13, LINE)
    c.showPage()


def answering():
    header(3, "用户问答：先检查版本与缓存，再执行 RAG", "缓存命中绕过模型推理；没有可用资料时提前结束。")
    box(50, 160, "1. 接收用户问题", ("python main.py ask <question>", "rag_demo/pipeline.py"))
    decision(455, 150, "Redis ready 存在？", "资料是否成功导入")
    box(860, 160, "2. 计算缓存键并查询", ("资料版本 + 问题 + 模型 + 配置", "SHA-256 / Redis GET"))
    arrow([(330, 205), (455, 205)])
    arrow([(735, 205), (860, 205)], "是", (787, 196))
    box(455, 285, "提示先成功执行 ingest", ("结束，不检索或生成回答",), RED, h=65)
    arrow([(595, 260), (595, 285)], "否", (609, 280))
    decision(860, 395, "缓存命中？", "命中直接返回既有结果")
    arrow([(1000, 250), (1000, 395)])
    box(455, 405, "3. 检索与精排", ("retrieve(question)", "详细流程见第 4 页"))
    arrow([(860, 450), (735, 450)], "否", (785, 438))
    decision(50, 395, "候选为空？", "Milvus 命中或回表结果为空")
    arrow([(455, 450), (410, 450), (410, 365), (190, 365), (190, 395)])
    box(50, 620, "提示没有检索到资料", ("结束，不调用 LLM / 不缓存",), RED, h=90)
    arrow([(190, 505), (190, 620)], "是", (205, 567))
    box(455, 620, "4. 组织引用并生成回答", ("编号 + 文件名 + 页码 + 原文", "ChatOpenAI 仅依据资料回答"))
    arrow([(330, 450), (382, 450), (382, 570), (595, 570), (595, 620)], "否", (401, 560))
    box(860, 620, "5. 输出答案与引用原文", ("未命中：先 SETEX 缓存 3600 秒", "已命中：直接输出缓存内容"), GREEN)
    arrow([(735, 665), (860, 665)])
    arrow([(1000, 505), (1000, 620)], "是", (1015, 567))
    text(55, 758, "缓存键随资料版本与参数变化；模型调用或 Redis 操作异常向外抛出，当前没有自动降级流程。", 13, LINE)
    c.showPage()


def retrieval():
    header(4, "检索细节：双路召回、加权融合与模型精排", "WeightedRanker 融合检索分数；BGE reranker 重新阅读问题与原文，两者职责不同。")
    box(455, 145, "输入用户问题", ("rag_demo/retrieval.py",), h=70)
    for center in (190, 1000):
        arrow([(595, 215), (595, 247), (center, 247), (center, 280)])
    box(50, 280, "语义路：BGE-M3 dense", ("问题编码 → COSINE 检索", "Milvus dense：最多召回 10 条"))
    box(860, 280, "关键词路：原生 BM25", ("原始问题 → 分词 / BM25 检索", "Milvus sparse：最多召回 10 条"))
    for center in (190, 1000):
        arrow([(center, 370), (center, 395), (595, 395), (595, 420)])
    box(455, 420, "WeightedRanker 融合", ("归一化后加权：dense 0.6 / BM25 0.4", "最多保留 10 个候选 ID"), AMBER)
    box(455, 555, "MySQL 按 ID 回表", ("取回片段原文、来源与页码", "SQL 返回顺序不作为最终顺序"), GREEN)
    arrow([(595, 510), (595, 555)])
    box(860, 555, "BGE reranker 精排", ("CrossEncoder(question, text)", "得到逐对相关性分数"))
    arrow([(735, 600), (860, 600)])
    box(860, 690, "按精排分数取前 3 条", ("不足 3 条时保留实际数量",), GREEN, h=70)
    arrow([(1000, 645), (1000, 690)])
    box(455, 690, "返回最终参考片段", ("交回第 3 页组织上下文与生成",), GREEN, h=70)
    arrow([(860, 725), (735, 725)])
    box(50, 555, "关键区别", (
        "BGE-M3 的 learned sparse", "不等于本例的 BM25 sparse。",
        "双路由同次 hybrid_search 执行。", "空命中或空回表结果返回空列表。",
        "当前未设置精排分数阈值。",
    ), fill=BLUE, h=205)
    c.showPage()


if __name__ == "__main__":
    # 2. 顺序生成四页，每页只讲解一层，避免把所有细节挤在同一张图上。
    register_font()
    c = canvas.Canvas(str(OUTPUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("Zhiyan 最小 RAG 项目流程图")
    c.setAuthor("Zhiyan 教学项目")
    overview()
    ingestion()
    answering()
    retrieval()
    # 3. 保存为根目录 docs 下的最终 PDF，渲染检查后再提交到 GitHub。
    c.save()
    print(OUTPUT)
