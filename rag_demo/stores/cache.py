"""Redis 连接；问答缓存与导入就绪版本。"""

import os
from functools import lru_cache

@lru_cache
def redis_client():
    from redis import Redis

    # URL 包含地址、端口和数据库号；解码后 get 直接返回字符串而非 bytes。
    return Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
                          decode_responses=True)

