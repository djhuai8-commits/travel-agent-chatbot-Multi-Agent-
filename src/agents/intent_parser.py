# ============================================================
# 旅行行程规划系统 - Multi-Agent RAG
# 意图解析 Agent
# ============================================================

import json
from typing import Optional
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


# ============================================================
# 意图槽位定义
# ============================================================

TRAVEL_INTENT_SCHEMA = {
    "origin": "出发城市（用户未明确则留空）",
    "destinations": "目的地城市/地区列表（按访问顺序）",
    "days_per_destination": "每个目的地的停留天数",
    "travelers": "出行人数",
    "companion_type": "出行人群类型（情侣/家庭/朋友/独自/商务）",
    "preferences": "用户偏好（美食/拍照/自然风光/历史文化/购物/休闲）",
    "dislikes": "用户不喜欢的内容（暴走/爬山/人挤人/购物团）",
    "pace": "行程节奏（轻松/适中/紧凑）",
    "budget": "预算区间（经济/中等/豪华），用户未提及则留空",
    "dietary_restrictions": "饮食限制（清真/素食/过敏等），无则留空"
}


# ============================================================
# Few-shot 示例
# ============================================================

FEW_SHOT_EXAMPLES = [
    {
        "input": "带爸妈去成都玩三天，都60多岁了，不想爬山",
        "output": json.dumps({
            "origin": "待定",
            "destinations": ["成都"],
            "days_per_destination": {"成都": 3},
            "travelers": 3,
            "companion_type": "家庭（老年人）",
            "preferences": ["历史文化", "休闲", "美食"],
            "dislikes": ["爬山", "暴走", "高强度徒步"],
            "pace": "轻松",
            "budget": "",
            "dietary_restrictions": ""
        }, ensure_ascii=False, indent=2)
    },
    {
        "input": "北京出发，厦门3天然后福州1天，情侣出行，喜欢拍照",
        "output": json.dumps({
            "origin": "北京",
            "destinations": ["厦门", "福州"],
            "days_per_destination": {"厦门": 3, "福州": 1},
            "travelers": 2,
            "companion_type": "情侣",
            "preferences": ["拍照", "自然风光", "美食", "海边"],
            "dislikes": ["暴走", "购物团"],
            "pace": "适中",
            "budget": "",
            "dietary_restrictions": ""
        }, ensure_ascii=False, indent=2)
    }
]


# ============================================================
# IntentParser Agent
# ============================================================

class IntentParserAgent:
    """
    第一层 Agent：意图解析

    职责：将用户的自然语言旅行请求解析为结构化 JSON 槽位。
    - 抽取目的地列表及停留天数
    - 识别出行人群类型和偏好
    - 识别不喜欢的内容（反向约束）
    - 确定行程节奏和预算
    """

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        use_local: bool = True,
        local_model: str = "qwen2.5:7b"
    ):
        """
        Args:
            model_name: OpenAI 模型名称（use_local=False 时使用）
            temperature: 生成温度，0.0 保证输出稳定
            use_local: 是否使用本地 Ollama 模型
            local_model: Ollama 本地模型名称
        """
        self._build_prompt_template()
        self._init_llm(model_name, temperature, use_local, local_model)

    def _build_prompt_template(self):
        """构建 System Prompt 和 Few-shot 模板"""

        schema_lines = "\n".join(
            f"- {k}: {v}" for k, v in TRAVEL_INTENT_SCHEMA.items()
        )

        few_shot_text = ""
        for ex in FEW_SHOT_EXAMPLES:
            few_shot_text += f"\n用户输入：{ex['input']}\n\n解析结果：\n{ex['output']}\n"

        self.prompt = PromptTemplate(
            template=(
                "你是一位专业的旅行规划助手，擅长从用户的自然语言中提取结构化的旅行需求。\n\n"
                "【输出格式说明】\n"
                f"{schema_lines}\n\n"
                "【输出要求】\n"
                "- 仅输出 JSON，不要有其他文字\n"
                "- 字段值尽量具体，不要留太多空字符串\n"
                "- destinations 按访问顺序排列\n"
                "- days_per_destination 键名必须和 destinations 中的城市名完全一致\n\n"
                "【Few-shot 示例】\n"
                f"{few_shot_text}"
                "现在开始解析用户请求：\n\n"
                "用户输入：{user_input}\n\n"
                "解析结果（JSON）："
            ),
            input_variables=["user_input"]
        )

    def _init_llm(self, model_name: str, temperature: float, use_local: bool, local_model: str):
        """初始化 LLM"""

        if use_local:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=local_model,
                    temperature=temperature,
                    format="json",
                    base_url="http://localhost:11434"
                )
            except ImportError:
                print("⚠️  未安装 langchain-ollama，回退到 OpenAI")
                use_local = False

        if not use_local:
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature
            )

        self.chain = LLMChain(prompt=self.prompt, llm=self.llm)

    def parse(self, user_input: str) -> dict:
        """
        解析用户输入，返回结构化意图 JSON。

        Args:
            user_input: 用户自然语言输入

        Returns:
            dict: 结构化意图字典

        Raises:
            ValueError: 解析失败或输出不是合法 JSON
        """
        raw = self.chain.invoke({"user_input": user_input})
        raw_text = raw["text"] if isinstance(raw, dict) else raw

        # 提取 JSON 部分（防止 LLM 输出多余文字）
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            intent = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Intent parsing failed: {e}\nRaw output: {text}")

        # 基本校验
        if "destinations" not in intent or not intent["destinations"]:
            raise ValueError("Parsing failed: no destinations extracted")

        return intent

    def parse_to_string(self, user_input: str) -> str:
        """返回格式化的 JSON 字符串（用于调试展示）"""
        return json.dumps(self.parse(user_input), ensure_ascii=False, indent=2)


# ============================================================
# 快速测试
# ============================================================

if __name__ == "__main__":
    agent = IntentParserAgent(use_local=True)

    test_cases = [
        "我和女朋友想去云南玩5天，喜欢拍照和美食，不想太累",
        "带爸妈去成都玩三天，都60多岁了，不想爬山",
        "北京出发，厦门3天然后福州1天，情侣出行，喜欢拍照",
    ]

    for text in test_cases:
        print(f"\n输入：{text}")
        print(f"输出：{agent.parse_to_string(text)}")
