"""AIFitBot 压力测试——模拟 100 人并发使用。

用法:
    locust -f tests/loadtest/locustfile.py --host=http://localhost:9000
    然后打开 http://localhost:8089 设置并发数和爬坡速率
"""

import json
import random
import time
from pathlib import Path

from locust import HttpUser, task, between, events


# ============================================================
# 加载测试数据
# ============================================================
_data_path = Path(__file__).parent / "test_data.json"
_TEST_DATA = None


def load_data():
    global _TEST_DATA
    if _TEST_DATA is None:
        with open(_data_path, "r", encoding="utf-8") as f:
            _TEST_DATA = json.load(f)
    return _TEST_DATA


# ============================================================
# 自定义事件：上报各环节耗时
# ============================================================
@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    """压测开始时输出提示。"""
    print("\n" + "=" * 60)
    print("AIFitBot 压力测试启动")
    print("=" * 60)
    print("注意: 压测会产生阿里云 API 调用费用！")
    print("建议先 10 人小流量验证\n")


class FitnessUser(HttpUser):
    """模拟一个健身用户的完整行为链路。"""

    # 每个用户操作间隔 3-8 秒（模拟真人打字+阅读）
    wait_time = between(3, 8)

    def on_start(self):
        """用户上线——随机取一个测试账号登录。"""
        data = load_data()
        user = random.choice(data["users"])
        self.token = user["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.session_id = 0  # 会在新建会话后更新

    # ============================================================
    # 任务定义
    # ============================================================

    @task(40)
    def chat_send(self):
        """核心任务：发送健身问题，触发完整 RAG 管道。

        这是最重的操作——含多 Query 扩展 + 向量检索 + BM25 + RRF + HyDE + rerank + LLM 生成。
        """
        data = load_data()
        question = random.choice(data["fitness_questions"])
        start = time.time()
        with self.client.post(
            "/chat/send",
            data={"question": question, "session_id": self.session_id},
            headers=self.headers,
            name="/chat/send [RAG]",
            catch_response=True,
        ) as resp:
            latency = time.time() - start
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
            elif "暂无相关信息" in resp.text:
                # 相关查询被误杀——质量门控过严
                resp.failure("False reject: quality gate rejected valid query")
            elif latency > 60:
                resp.failure(f"Timeout: {latency:.1f}s")
            else:
                resp.success()

    @task(15)
    def chat_send_irrelevant(self):
        """次要任务：发送无关问题，应触发拒答（轻量，不走 LLM 生成）。"""
        data = load_data()
        question = random.choice(data["irrelevant_questions"])
        with self.client.post(
            "/chat/send",
            data={"question": question, "session_id": self.session_id},
            headers=self.headers,
            name="/chat/send [reject]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
            elif "暂无相关信息" not in resp.text:
                # 无关查询没被拒答——质量门控过松
                resp.failure("False pass: irrelevant query not rejected")
            else:
                resp.success()

    @task(10)
    def chat_stream(self):
        """流式任务：SSE 流式输出，逐 token 推送。"""
        data = load_data()
        question = random.choice(data["fitness_questions"])
        start = time.time()
        with self.client.post(
            "/chat/stream",
            data={"question": question, "session_id": self.session_id},
            headers=self.headers,
            name="/chat/stream [SSE]",
            catch_response=True,
            stream=True,
        ) as resp:
            # 流式需要消费完整个响应
            done = False
            for line in resp.iter_lines():
                if line and b'"type":"done"' in line:
                    done = True
            latency = time.time() - start
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
            elif not done:
                resp.failure(f"Stream incomplete after {latency:.1f}s")
            elif latency > 90:
                resp.failure(f"Stream timeout: {latency:.1f}s")
            else:
                resp.success()

    @task(5)
    def session_list(self):
        """轻量任务：获取会话列表（纯 DB 查询）。"""
        with self.client.get(
            "/chat/session/list",
            headers=self.headers,
            name="/chat/session/list [DB]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
            else:
                resp.success()

    @task(10)
    def health_check(self):
        """极轻任务：健康检查端点基准。"""
        with self.client.get(
            "/api/health",
            name="/api/health",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
            else:
                resp.success()

    @task(5)
    def new_session(self):
        """新建会话。"""
        with self.client.post(
            "/chat/session/new",
            data={"title": f"压测会话-{random.randint(1,999)}"},
            headers=self.headers,
            name="/chat/session/new",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                self.session_id = data.get("session_id", 0)
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")
