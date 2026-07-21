"""压测数据准备——直接生成 JWT token，不经过 HTTP 登录，秒级完成。"""

import json
import sys
import io
from pathlib import Path
from datetime import datetime, timedelta, timezone

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jose import jwt
from src.config import JWT_SECRET, JWT_ALGORITHM
from src.database import engine
from src.services.auth_service import AuthService
from src.models.user import User
from sqlmodel import Session, select

USER_COUNT = 100
PASSWORD = "loadtest123"

FITNESS_QUESTIONS = [
    "杠铃卧推标准动作要领是什么",
    "怎么练出六块腹肌",
    "减脂期饮食应该怎么安排",
    "深蹲膝盖疼怎么办",
    "二头肌怎么练大",
    "蛋白粉什么时候喝效果最好",
    "新手一周练几次合适",
    "硬拉的标准动作",
    "拉伸对增肌有帮助吗",
    "练胸最好的动作有哪些",
    "减脂和增肌可以同时进行吗",
    "训练后肌肉酸痛怎么缓解",
    "女生练肩会变壮吗",
    "碳水循环怎么操作",
    "俯卧撑练哪里",
    "引体向上一个都做不了怎么办",
    "肌酸怎么吃",
    "空腹训练好吗",
    "核心训练有哪些动作",
    "三角肌后束怎么练",
]

IRRELEVANT_QUESTIONS = [
    "今天天气怎么样",
    "2024年世界杯冠军是谁",
    "股票怎么买赚钱",
    "Python列表推导式怎么写",
    "推荐一部好看的电影",
]

print(f"批量创建 {USER_COUNT} 个测试用户（直接注册 + 生成 JWT）...")

tokens = []
with Session(engine) as session:
    auth = AuthService(session)
    for i in range(1, USER_COUNT + 1):
        username = f"loadtest_{i:03d}"
        # 注册（已存在则跳过）
        try:
            auth.register(username, PASSWORD)
        except Exception:
            pass

        # 查用户 ID，生成 JWT（sub 必须是 user_id 整数）
        user = session.exec(
            select(User).where(User.username == username)
        ).first()
        if not user:
            print(f"  {username}: 注册失败")
            continue

        expire = datetime.now(timezone.utc) + timedelta(hours=24)
        payload = {"sub": str(user.id), "exp": expire}
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        tokens.append({"username": username, "token": token})

        if i % 20 == 0:
            print(f"  进度: {i}/{USER_COUNT}")

output = {
    "users": tokens,
    "fitness_questions": FITNESS_QUESTIONS,
    "irrelevant_questions": IRRELEVANT_QUESTIONS,
}
out_path = Path(__file__).parent / "test_data.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"完成: {len(tokens)} 个用户已生成")
print(f"数据保存到 {out_path}")
