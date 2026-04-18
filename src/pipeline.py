# ============================================================
# 旅行行程规划系统 - Multi-Agent RAG
# 主流程编排管道
# ============================================================

import json
import time
from typing import Optional, Dict, Any
from pathlib import Path

from .agents.intent_parser import IntentParserAgent
from .agents.knowledge_retriever import KnowledgeRetrieverAgent, HybridRetriever, format_retrieval_context
from .agents.itinerary_planner import ItineraryPlannerAgent


class TravelAgentPipeline:
    """
    三层 Agent 串联主流程

    Flow:
    用户输入
      ↓
    [Agent 1: IntentParser] → 结构化意图 JSON
      ↓
    [Agent 2: KnowledgeRetriever] → 各目的地检索结果
      ↓
    [Agent 3: ItineraryPlanner] → 每日行程安排
      ↓
    返回最终行程
    """

    def __init__(
        self,
        hybrid_retriever: Optional[HybridRetriever] = None,
        use_local: bool = True,
        use_hyde: bool = True,
        local_model: str = "qwen2.5:7b",
        embed_model: str = "nomic-embed-text"
    ):
        """
        Args:
            hybrid_retriever: 已构建好的混合检索器，如不提供则自动构建
            use_local: 是否使用本地 Ollama 模型
            use_hyde: 是否启用 HyDE 冷查询优化
            local_model: Ollama 本地模型名称
            embed_model: 嵌入模型名称
        """
        self.use_hyde = use_hyde

        # 初始化三层 Agent
        self.intent_agent = IntentParserAgent(use_local=use_local, local_model=local_model)
        self.know_agent = KnowledgeRetrieverAgent(
            hybrid_retriever=hybrid_retriever,
            use_hyde=use_hyde,
            use_local=use_local,
            local_model=local_model
        )
        self.planner_agent = ItineraryPlannerAgent(use_local=use_local, local_model=local_model)

        self._metrics: Dict[str, Any] = {}

    def run(self, user_input: str, verbose: bool = False) -> Dict[str, Any]:
        """
        执行完整的三层 Agent 流程

        Args:
            user_input: 用户自然语言旅行需求
            verbose: 是否打印各层中间结果

        Returns:
            {
                "intent": 意图解析结果,
                "retrieval_results": {目的地: [结果列表]},
                "itinerary": 最终行程文本,
                "timing": 各阶段耗时,
                "metrics": 关键指标
            }
        """
        start_total = time.time()
        timing = {}

        # ---- Layer 1: 意图解析 ----
        t0 = time.time()
        intent = self.intent_agent.parse(user_input)
        timing["intent_parsing"] = time.time() - t0
        if verbose:
            print(f"\n🧠 [Layer 1] 意图解析: {timing['intent_parsing']:.2f}s")
            print(f"   目的地: {intent['destinations']}")
            print(f"   偏好: {intent.get('preferences', [])}")

        # ---- Layer 2: 知识检索 ----
        t0 = time.time()
        retrieval_results = self.know_agent.batch_retrieve(intent)
        timing["knowledge_retrieval"] = time.time() - t0
        knowledge_context = format_retrieval_context(retrieval_results)
        if verbose:
            print(f"\n📚 [Layer 2] 知识检索: {timing['knowledge_retrieval']:.2f}s")
            total_docs = sum(len(v) for v in retrieval_results.values())
            print(f"   共检索 {total_docs} 个文档块")

        # ---- Layer 3: 行程编排 ----
        t0 = time.time()
        itinerary = self.planner_agent.plan_with_retry(intent, knowledge_context)
        timing["itinerary_planning"] = time.time() - t0
        if verbose:
            print(f"\n🗺️  [Layer 3] 行程编排: {timing['itinerary_planning']:.2f}s")

        timing["total"] = time.time() - start_total

        # 记录指标
        self._metrics = {
            "total_time": timing["total"],
            "intent_parsing_time": timing["intent_parsing"],
            "retrieval_time": timing["knowledge_retrieval"],
            "planning_time": timing["itinerary_planning"],
            "num_destinations": len(intent.get("destinations", [])),
            "total_retrieved_docs": sum(len(v) for v in retrieval_results.values()),
        }

        if verbose:
            print(f"\n⏱️  总耗时: {timing['total']:.2f}s")

        return {
            "intent": intent,
            "retrieval_results": retrieval_results,
            "itinerary": itinerary,
            "timing": timing,
            "metrics": self._metrics
        }

    def run_stream(self, user_input: str):
        """
        流式版本（当前实现为简化版非真正流式）
        实际使用时可配合 langchain StreamingChatModel
        """
        result = self.run(user_input, verbose=False)
        yield result

    def get_metrics(self) -> Dict[str, Any]:
        """获取最近一次运行的性能指标"""
        return self._metrics


# ============================================================
# 工具函数：从文件构建检索器
# ============================================================

def build_retriever_from_json(
    json_path: str,
    embed_model: str = "nomic-embed-text",
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> HybridRetriever:
    """
    从 JSON 文件加载旅行指南数据，构建混合检索器

    Args:
        json_path: JSON 文件路径，格式：
            [{"destination": "云南", "content": "...", "tags": [...]}, ...]
        embed_model: 嵌入模型
        chunk_size: chunk 大小（tokens，近似字符数/2）
        chunk_overlap: overlap 字符数
    """
    from llama_index.core import Document

    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 构建 LlamaIndex Document
    documents = []
    for item in raw_data:
        meta = {
            "destination": item.get("destination", ""),
            "tags": ",".join(item.get("tags", [])),
            "source": item.get("source", "unknown")
        }
        doc = Document(text=item["content"], metadata=meta)
        documents.append(doc)

    retriever = HybridRetriever(embed_model_name=embed_model)
    retriever.build(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return retriever


# ============================================================
# 快速演示
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="我想去云南玩5天，情侣出行，喜欢拍照和美食")
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    # 构建检索器
    data_path = Path(__file__).parent.parent / "data" / "sample_travel_guides.json"
    if data_path.exists() and args.rebuild_index:
        print("🔧 构建知识库索引...")
        retriever = build_retriever_from_json(str(data_path))
    else:
        print("⚠️  未找到数据文件，跳过索引构建（请先准备 data/sample_travel_guides.json）")
        retriever = None

    # 构建管道
    pipeline = TravelAgentPipeline(hybrid_retriever=retriever, use_local=True)

    # 运行
    result = pipeline.run(args.input, verbose=True)
    print("\n" + "=" * 60)
    print("📋 最终行程")
    print("=" * 60)
    print(result["itinerary"])
