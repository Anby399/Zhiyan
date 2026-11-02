"""MySQL 片段与来源存储。"""

import os
from functools import lru_cache

@lru_cache
def mysql():
    import pymysql

    db = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "rag"),
        password=os.getenv("MYSQL_PASSWORD", "rag_demo_password"),
        database=os.getenv("MYSQL_DATABASE", "rag_demo"),
        charset="utf8mb4", autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    with db.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS teaching_rag_chunks (
            id VARCHAR(64) PRIMARY KEY, source TEXT NOT NULL,
            page INT NOT NULL, text MEDIUMTEXT NOT NULL)""")
    return db

