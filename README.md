# Travel Agent Chatbot - Multi-Agent RAG Itinerary Planner

基于 Multi-Agent 架构与 RAG 的智能旅行行程规划系统，使用 LangChain + LlamaIndex 实现三层 Agent 串行协作。

## 项目背景

单一 Agent 处理复杂跨城多日行程时，上下文过长导致模型注意力分散，行程逻辑错误率高。本项目将规划任务拆解为三层 Agent：
**意图解析 Agent → 知识检索 Agent → 行程编排 Agent**，各司其职，显著降低复杂请求的错误率。

## 核心特性

- **三层 Multi-Agent 架构**：意图解析（槽位抽取）→ 知识检索（RAG）→ 行程编排（结构化输出）
- **混合检索**：向量检索 + BM25 关键词召回，融合 Rerank
- **HyDE 支持**：假设文档嵌入优化小众目的地冷查询召回
- **自动化评测**：100 条标注测试集，覆盖简单单城 / 跨城多日 / 含特殊偏好三类
- **Gradio 可视化界面**：多轮对话、行程修改、追问延伸

## 效果数据

| 指标 | 单 Agent 基线 | 三层 Agent 方案 | 提升 |
|------|-------------|----------------|------|
| 行程逻辑正确率 | 68% | 89% | +21 pct |
| Top-3 检索命中率 | 73% | 91% | +18 pct |
| 平均响应时间 | 4.2s | 3.5s | -17% |

## 技术栈

`Python 3.10+` `LangChain` `LlamaIndex` `ChromaDB` `FAISS` `sentence-transformers` `Gradio`

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key 或使用本地 Ollama
```

如使用本地模型（推荐，无需 API Key）：

```bash
# 安装 Ollama: https://ollama.com/download
ollama pull qwen2.5:7b
ollama pull bge-m3:latest
```

### 3. 构建知识库

```bash
python run_demo.py --rebuild-index
```

### 4. 启动 Gradio 界面

```bash
python gradio_app.py
```

### 5. 运行评测

```bash
python run_eval.py
```

## 项目结构

```
travel-agent-chatbot/
├── src/
│   ├── config.py              # 全局配置
│   ├── pipeline.py            # 三层 Agent 串联主流程
│   ├── evaluation.py           # 评测脚本
│   └── agents/
│       ├── intent_parser.py   # 意图解析 Agent
│       ├── knowledge_retriever.py  # 知识检索 Agent (RAG)
│       └── itinerary_planner.py    # 行程编排 Agent
├── data/
│   └── sample_travel_guides.json   # 示例旅行攻略数据
├── gradio_app.py              # Gradio 可视化界面
├── run_demo.py                # 命令行演示脚本
├── run_eval.py                # 评测脚本
└── requirements.txt
```

## Agent 架构详解

### 第一层：意图解析 Agent

**职责**：将自然语言输入解析为结构化 JSON。

输入示例：
> "我和女朋友想去云南玩5天，喜欢拍照和美食，不想太累"

输出：
```json
{
  "origin": "待定",
  "destinations": ["云南"],
  "days_per_destination": {"云南": 5},
  "travelers": 2,
  "companion_type": "情侣",
  "preferences": ["拍照", "美食"],
  "dislikes": ["暴走", "爬山"],
  "pace": "轻松"
}
```

### 第二层：知识检索 Agent

**职责**：对每个目的地，分别从 RAG 知识库召回相关攻略、景点、酒店、餐饮信息。

核心流程：
1. BM25 稀疏检索（关键词匹配兜底）
2. 向量检索（语义召回）
3. RRF 融合两路结果
4. BGE-Reranker 重排，输出 Top-5

### 第三层：行程编排 Agent

**职责**：综合意图 + 检索结果，生成结构化每日行程。

输出格式：
```
Day 1（昆明）
- 上午：滇池海埂公园（免费观鸥，拍照绝佳）★★★★☆
- 下午：翠湖公园 → 云南大学附近闲逛
- 晚上：昆明老街（南强街）品尝过桥米线、汽锅鸡
```

## 实验记录

详见 `docs/experiments.md`，包含：
- 5 种 Chunk Size 对比（256/512/768/1024/1536 tokens）
- 4 种检索策略对比
- Prompt 模板调优过程
- Few-shot 示例选择依据

## License

MIT
