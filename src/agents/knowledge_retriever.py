# ============================================================
# 旅行行程规划系统 - Multi-Agent RAG
# 知识检索 Agent (RAG)
# ============================================================

import json
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor
from llama_index.core.schema import NodeWithScore
try:
    from llama_index.postprocessor.cohere_rerank import CohereRerank
except Exception:
    CohereRerank = None

import faiss
import numpy as np


# ============================================================
# 配置
# ============================================================

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_RERANK = 5


# ============================================================
# HybridRetriever - BM25 + 向量混合检索
# ============================================================

class HybridRetriever:
    """
    混合检索器：融合 BM25 稀疏检索与向量检索

    采用 Reciprocal Rank Fusion (RRF) 合并两路结果：
    RRF_score(d) = Σ 1/(k+rank_i(d))
    k=60（标准参数），k 越小越偏向"两路都靠前"的文档

    支持 Ollama 本地 embedding（默认）和 SentenceTransformer 两种后端。
    """

    def __init__(
        self,
        embed_model_name: str = "nomic-embed-text",
        k: int = 60,
        device: str = "cpu",
        ollama_base_url: str = "http://localhost:11434",
        use_ollama_embed: bool = True
    ):
        self.k = k
        self.use_ollama_embed = use_ollama_embed
        self.embedding_dim = None

        if self.use_ollama_embed:
            # 使用 Ollama 本地 embedding（无需下载 HuggingFace 模型）
            import requests
            self._ollama_url = f"{ollama_base_url.rstrip('/')}/api/embeddings"
            self._model_name = embed_model_name
            self._session = requests.Session()
            # 先探测维度
            try:
                test_resp = self._session.post(self._ollama_url, json={
                    "model": self._model_name, "prompt": "test"
                }, timeout=120)
                test_resp.raise_for_status()
                vec = test_resp.json().get("embedding", [])
                self.embedding_dim = len(vec)
                print(f"✅ Ollama embedding 模型: {self._model_name}, 维度: {self.embedding_dim}")
            except Exception as e:
                print(f"⚠️ Ollama embedding 连接失败: {e}")
                raise
            self.embed_model = None  # 不需要 SentenceTransformer
        else:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(embed_model_name, device=device)
            self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.doc_metadata: List[Dict[str, Any]] = []
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._is_built = False

    def build(self, documents: List[Document], chunk_size: int = DEFAULT_CHUNK_SIZE,
              chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """构建 FAISS 索引和 BM25 检索器"""

        texts = [doc.text for doc in documents]
        self.documents = texts
        self.doc_metadata = [doc.metadata for doc in documents]

        # ---------- 向量索引 ----------
        if self.use_ollama_embed:
            # 使用 Ollama API 批量生成 embeddings（分批避免超时）
            batch_size = 16
            all_embeddings = []
            total = len(texts)
            for i in range(0, total, batch_size):
                batch = texts[i:i + batch_size]
                # 用 Ollama /api/embeddings 逐个请求（Ollama 不原生支持批量）
                batch_embs = []
                for text in batch:
                    resp = self._session.post(
                        self._ollama_url,
                        json={"model": self._model_name, "prompt": text},
                        timeout=120
                    )
                    resp.raise_for_status()
                    batch_embs.append(resp.json()["embedding"])
                all_embeddings.extend(batch_embs)
                print(f"  Embedding 进度: {min(i + batch_size, total)}/{total}")
            embeddings = np.array(all_embeddings, dtype=np.float32)
        else:
            embeddings = self.embed_model.encode(texts, show_progress_bar=True,
                                                  batch_size=32, convert_to_numpy=True)

        # 使用 IndexFlatIP（内积）做相似度检索
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype(np.float32))
        self._is_built = True

        # ---------- BM25 索引 ----------
        try:
            import jieba
            tokenized = [list(jieba.cut(text)) for text in texts]
        except ImportError:
            import re
            tokenized = [re.findall(r'\w+', text) for text in texts]

        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi(tokenized)
        self._tokenized_docs = tokenized

        print(f"✅ 索引构建完成：{len(texts)} 个文档块，维度 {self.embedding_dim}")

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """向量检索"""
        if self.use_ollama_embed:
            resp = self._session.post(
                self._ollama_url,
                json={"model": self._model_name, "prompt": query},
                timeout=120
            )
            resp.raise_for_status()
            q_emb = np.array([resp.json()["embedding"]], dtype=np.float32)
        else:
            q_emb = self.embed_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 稀疏检索"""
        try:
            import jieba
            q_tokens = list(jieba.cut(query))
        except ImportError:
            import re
            q_tokens = re.findall(r'\w+', query)

        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(int(idx), float(score)) for idx, score in ranked[:top_k]]

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[int, float]],
        bm25_results: List[Tuple[int, float]],
        top_k: int
    ) -> List[Tuple[int, float, str]]:
        """
        Reciprocal Rank Fusion 融合两路结果

        Args:
            vector_results: [(doc_idx, score), ...]
            bm25_results: [(doc_idx, score), ...]
            top_k: 最终返回数量

        Returns:
            [(doc_idx, rrf_score, source), ...]
        """
        rrf_scores: Dict[int, float] = {}

        for rank, (doc_idx, _) in enumerate(vector_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (self.k + rank + 1)

        for rank, (doc_idx, _) in enumerate(bm25_results):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (self.k + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, score in ranked[:top_k]:
            # 判断来源
            vec_ranks = [r for r, (i, _) in enumerate(vector_results) if i == doc_idx]
            bm25_ranks = [r for r, (i, _) in enumerate(bm25_results) if i == doc_idx]
            if vec_ranks and bm25_ranks:
                source = "hybrid"
            elif vec_ranks:
                source = "vector"
            else:
                source = "bm25"
            results.append((doc_idx, score, source))

        return results

    def retrieve(self, query: str, top_k: int = TOP_K_RERANK) -> List[Dict[str, Any]]:
        """
        执行混合检索，返回 Top-K 融合结果

        Returns:
            [{"text": str, "score": float, "source": str, "metadata": dict}, ...]
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build() first.")

        vector_results = self._vector_search(query, TOP_K_VECTOR)
        bm25_results = self._bm25_search(query, TOP_K_BM25)

        fused = self._rrf_fusion(vector_results, bm25_results, top_k)

        return [
            {
                "text": self.documents[idx],
                "score": score,
                "source": src,
                "metadata": self.doc_metadata[idx]
            }
            for idx, score, src in fused
        ]


# ============================================================
# KnowledgeRetriever Agent
# ============================================================

class KnowledgeRetrieverAgent:
    """
    第二层 Agent：知识检索

    对每个目的地分别执行混合 RAG 检索，召回相关攻略内容。
    支持 HyDE 模式：先用 LLM 生成假设答案，再以假设答案做检索，
    有效提升小众目的地冷查询的召回质量。
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        use_hyde: bool = True,
        model_name: str = "gpt-4o-mini",
        use_local: bool = True,
        local_model: str = "qwen2.5:7b"
    ):
        self.retriever = hybrid_retriever
        self.use_hyde = use_hyde
        self._init_llm(model_name, use_local, local_model)

    def _init_llm(self, model_name: str, use_local: bool, local_model: str):
        if use_local:
            try:
                from langchain_ollama import ChatOllama
                self.llm = ChatOllama(
                    model=local_model,
                    temperature=0.7,
                    base_url="http://localhost:11434"
                )
            except ImportError:
                use_local = False

        if not use_local:
            self.llm = ChatOpenAI(model=model_name, temperature=0.7)

    def _hyde_generate(self, query: str) -> str:
        """HyDE: 根据用户问题生成假设的"完美答案"，用于检索"""
        prompt = (
            "你是一位旅行攻略专家。请根据以下用户问题，"
            "生成一段假设的完美答案（包含目的地、景点、建议、注意事项等详细信息）。\n\n"
            f"用户问题：{query}\n\n"
            "假设的完美答案："
        )
        result = self.llm.invoke(prompt)
        return result.content if hasattr(result, 'content') else str(result)

    def _build_search_query(self, intent: dict, destination: str) -> str:
        """为目的地构建检索查询"""
        prefs = intent.get("preferences", [])
        dislikes = intent.get("dislikes", [])
        pace = intent.get("pace", "适中")
        days = intent.get("days_per_destination", {}).get(destination, 0)
        companion = intent.get("companion_type", "")

        parts = [f"{destination}旅游攻略"]
        if prefs:
            parts.append(f"偏好：{', '.join(prefs)}")
        if dislikes:
            parts.append(f"避免：{', '.join(dislikes)}")
        parts.append(f"行程{days}天，{companion}出行，节奏{pace}")
        return "，".join(parts)

    def retrieve_for_destination(
        self,
        intent: dict,
        destination: str,
        hyde_query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        为单个目的地执行知识检索

        Args:
            intent: 意图解析结果
            destination: 目标目的地
            hyde_query: 可选的用户原始问题（用于 HyDE 生成假设答案）

        Returns:
            检索结果列表，每项含 text/score/source/metadata
        """
        search_query = self._build_search_query(intent, destination)

        if self.use_hyde and hyde_query:
            # HyDE: 先生成假设答案，再做检索
            hypothetical = self._hyde_generate(hyde_query)
            retrieval_query = hypothetical
        else:
            retrieval_query = search_query

        results = self.retriever.retrieve(retrieval_query, top_k=TOP_K_RERANK)
        return results

    def batch_retrieve(self, intent: dict) -> Dict[str, List[Dict[str, Any]]]:
        """
        对意图中的所有目的地批量检索

        Returns:
            {目的地: [检索结果列表]}
        """
        destinations = intent.get("destinations", [])
        hyde_query = (
            f"{intent.get('preferences', [])}, "
            f"{intent.get('companion_type', '')}出行, "
            f"{' '.join(destinations)}旅行"
        )

        results_map = {}
        for dest in destinations:
            results_map[dest] = self.retrieve_for_destination(intent, dest, hyde_query)
            print(f"  📍 {dest}: 检索到 {len(results_map[dest])} 条结果")

        return results_map


# ============================================================
# 辅助函数：格式化检索结果
# ============================================================

def format_retrieval_context(results_map: Dict[str, List[Dict[str, Any]]]) -> str:
    """
    将检索结果格式化为行程编排 Agent 可消费的上下文文本
    """
    lines = []
    for dest, results in results_map.items():
        lines.append(f"\n{'='*20} {dest} 旅行攻略 {'='*20}\n")
        for i, r in enumerate(results, 1):
            source_tag = f"[来源:{r['source']}]"
            meta = r.get("metadata", {})
            location = meta.get("location", dest)
            lines.append(f"{i}. {source_tag} {location}\n{r['text']}\n")
    return "\n".join(lines)


if __name__ == "__main__":
    print("KnowledgeRetriever 模块直接测试")
    # 实际使用时需先构建索引，参考 pipeline.py
