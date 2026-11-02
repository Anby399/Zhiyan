"""MySQL 片段与来源存储。"""

import os
from functools import lru_cache

# 同一串行命令内复用连接；本示例不实现连接池。
@lru_cache
def mysql():
    import pymysql

    # 1. 从 .env 已加载的环境变量连接数据库，DictCursor 按字段名返回结果。
    db = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "rag"),
        password=os.getenv("MYSQL_PASSWORD", "rag_demo_password"),
        database=os.getenv("MYSQL_DATABASE", "rag_demo"),
        charset="utf8mb4", autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    # 2. 只负责建表；数据库与账号由 Compose 初始化，原文和向量通过 id 关联。
    with db.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS teaching_rag_chunks (
            id VARCHAR(64) PRIMARY KEY, source TEXT NOT NULL,
            page INT NOT NULL, text MEDIUMTEXT NOT NULL)""")
    return db

