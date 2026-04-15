# ============================================================
# 旅行行程规划系统 - Multi-Agent RAG
# 行程编排 Agent
# ============================================================

import json
from typing import Dict, Any, List, Optional
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


# ============================================================
# Few-shot 示例（正例 + 反例修正）
# ============================================================

FEW_SHOT_ITINERARY = {
    "positive": [
        {
            "context": (
                "目的地：云南（丽江+大理），5天，情侣出行，偏好拍照和美食，节奏轻松，避免暴走\n\n"
                "检索内容摘要：\n"
                "- D1丽江：丽江古城（大研古镇）夜景观赏，忠义市场品尝腊排骨火锅\n"
                "- D2丽江：束河古镇（比大研安静，适合拍照），玉龙雪山云杉坪（轻松索道）\n"
                "- D3-D4大理：洱海西岸骑行（才村-喜洲），双廊海景民宿，大理古城\n"
                "- D5大理/返程：苍山感通索道（或洗马潭），下午返程"
            ),
            "output": (
                "**云南5天4晚情侣轻奢游**\n\n"
                "**Day 1｜丽江抵达**\n"
                "🌙 晚上：丽江古城（大研）闲逛，忠义市场品尝腊排骨火锅\n"
                "📸 推荐：狮子山万古楼夜景（俯瞰古城全景）\n\n"
                "**Day 2｜丽江：玉龙雪山（轻松版）**\n"
                "☀️ 上午：玉龙雪山云杉坪（索道直达，景色开阔不需爬山）\n"
                "🍜 中午：束河古镇午餐，尝纳西烤鱼\n"
                "🌿 下午：束河古镇慢逛，适合情侣拍照，人少安静\n"
                "🏨 住宿：束河古镇内客栈\n\n"
                "**Day 3｜丽江→大理**\n"
                "🚄 上午：乘火车前往大理（约2小时）\n"
                "🏠 下午：入住双廊海景民宿（洱海东岸，看日落绝佳）\n"
                "🌅 傍晚：双廊洱海边散步，品尝银米米线\n\n"
                "**Day 4｜大理：洱海骑行**\n"
                "🚴 上午：双廊→喜洲洱海骑行（约20km，推荐租电动车或自行车）\n"
                "🏛️ 下午：喜洲古镇（严家大院），品尝喜洲粑粑\n"
                "🌙 晚上：返回双廊，海景餐厅晚餐\n\n"
                "**Day 5｜大理：苍山游览**\n"
                "☀️ 上午：苍山感通索道（俯瞰洱海，体力要求低）\n"
                "🍜 中午：大理古城午餐，尝凉鸡米线\n"
                "🚄 下午：根据返程交通安排，提前1.5小时出发前往机场/火车站\n\n"
                "---\n"
                "💡 小贴士：防晒！云南紫外线强；洱海骑行建议上午出发，避免下午逆光"
            )
        }
    ],
    "negative": {
        "bad_output": (
            "Day 1: 玉龙雪山（爬山）\n"
            "Day 2: 丽江古城\n"
            "Day 3: 双廊\n"
            "Day 4: 爬苍山\n"
            "Day 5: 再去一次玉龙雪山\n"
            "（问题：行程节奏混乱，同类景点重复，爬山路不符合用户轻松偏好）"
        ),
        "correction": (
            "Day 1: 抵达丽江，游览古城夜景\n"
            "Day 2: 云杉坪（索道，轻松）\n"
            "Day 3: 火车去大理\n"
            "Day 4: 洱海骑行\n"
            "Day 5: 苍山索道游览，返程\n"
            "（修正：节奏从难到易，避免爬山类景点，同一天就近安排）"
        )
    }
}


# ============================================================
# ItineraryPlanner Agent
# ============================================================

class ItineraryPlannerAgent:
    """
    第三层 Agent：行程编排

    综合意图 JSON + 检索到的目的地知识，生成结构化每日行程。
    通过 Few-shot（正例+反例）教会模型：
    - 避免节奏混乱（同一天远距离景点折返）
    - 避免重复类似景点
    - 优先满足用户偏好，同时规避不喜欢的活动
    - 合理安排出行时间，留足缓冲
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        use_local: bool = True,
        local_model: str = "qwen2.5:7b"
    ):
        self._build_prompt_template()
        self._init_llm(model_name, temperature, use_local, local_model)

    def _build_prompt_template(self):
        """构建行程编排 Prompt"""

        few_shot_text = "\n".join([
            f"\n示例 {i+1}：\n\n背景信息：\n{ex['context']}\n\n"
            f"行程安排：\n{ex['output']}\n"
            for i, ex in enumerate(FEW_SHOT_ITINERARY["positive"])
        ])

        neg = FEW_SHOT_ITINERARY["negative"]

        self.prompt = PromptTemplate(
            template=(
                "你是一位专业的旅行规划师，擅长根据用户需求和目的地信息，\n"
                "生成结构清晰、逻辑合理、符合用户偏好的每日行程安排。\n\n"
                "【行程编排规则】（必须遵守）\n"
                "1. 同一城市内每天景点应地理位置相近，避免长时间折返\n"
                "2. 避免安排用户明确不喜欢的活动类型\n"
                "3. 节奏：轻松偏好→每天不超过2个大景点；紧凑偏好→可排3-4个\n"
                "4. 每个景点注明推荐理由（用emoji）和大致游览时长\n"
                "5. 首尾天需考虑到达/离开的时间约束（交通、用餐）\n"
                "6. 最后一天不要安排远距离景点，留足返程时间\n"
                "7. 每天行程包含：上午/下午/晚上各时段安排，附餐厅推荐\n\n"
                "【负面示例】（必须避免的错误）\n"
                f"❌ 错误示范：\n{neg['bad_output']}\n\n"
                f"✅ 正确修正：\n{neg['correction']}\n\n"
                "【参考示例】\n"
                f"{few_shot_text}\n\n"
                "【用户旅行需求（JSON）】\n"
                "{intent_json}\n\n"
                "【目的地检索知识】\n"
                "{knowledge_context}\n\n"
                "请按上述规则生成完整行程安排："
            ),
            input_variables=["intent_json", "knowledge_context"]
        )

    def _init_llm(self, model_name: str, temperature: float, use_local: bool, local_model: str):
        if use_local:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=local_model,
                    temperature=temperature,
                    base_url="http://localhost:11434"
                )
            except ImportError:
                use_local = False

        if not use_local:
            self.llm = ChatOpenAI(model=model_name, temperature=temperature)

        self.chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def plan(
        self,
        intent: dict,
        knowledge_context: str
    ) -> str:
        """
        生成行程规划

        Args:
            intent: 意图解析 JSON
            knowledge_context: 格式化后的检索结果文本

        Returns:
            行程规划文本
        """
        result = self.chain.invoke({
            "intent_json": json.dumps(intent, ensure_ascii=False, indent=2),
            "knowledge_context": knowledge_context
        })

        text = result["text"] if isinstance(result, dict) else result
        return text.strip()

    def plan_with_retry(
        self,
        intent: dict,
        knowledge_context: str,
        max_retries: int = 2
    ) -> str:
        """带重试的行程规划（检验格式合理性）"""
        for attempt in range(max_retries + 1):
            result = self.plan(intent, knowledge_context)
            # 简单格式检验：至少包含 "Day" 关键词
            if "Day" in result or "day" in result:
                return result
            if attempt < max_retries:
                print(f"⚠️  格式检验失败，重试第 {attempt + 2} 次...")
        return result


if __name__ == "__main__":
    agent = ItineraryPlannerAgent(use_local=True)

    test_intent = {
        "origin": "上海",
        "destinations": ["成都"],
        "days_per_destination": {"成都": 3},
        "travelers": 2,
        "companion_type": "情侣",
        "preferences": ["美食", "历史文化"],
        "dislikes": ["爬山"],
        "pace": "适中",
        "budget": "中等",
        "dietary_restrictions": ""
    }

    test_context = (
        "成都三日游攻略摘要：\n"
        "- D1：宽窄巷子（免费历史街区）、锦里（夜景）\n"
        "- D2：熊猫基地（早上去看熊猫）、武侯祠（了解三国文化）\n"
        "- D3：都江堰（世界文化遗产，半天足够）或青城山（爬山）\n"
        "- 美食推荐：玉林串串、蜀大侠火锅、降龙爪爪"
    )

    itinerary = agent.plan_with_retry(test_intent, test_context)
    print(itinerary)
