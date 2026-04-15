# ============================================================
# 旅行行程规划系统 - Gradio 可视化界面
# ============================================================

import gradio as gr
from src.pipeline import TravelAgentPipeline, build_retriever_from_json
from pathlib import Path


# 全局管道实例（延迟初始化）
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        data_path = Path(__file__).parent / "data" / "sample_travel_guides.json"
        if data_path.exists():
            print("🔧 加载知识库...")
            retriever = build_retriever_from_json(str(data_path))
            _pipeline = TravelAgentPipeline(hybrid_retriever=retriever, use_local=True)
        else:
            print("⚠️  未找到数据文件，使用无检索模式（行程基于 LLM 生成）")
            _pipeline = TravelAgentPipeline(hybrid_retriever=None, use_local=True)
    return _pipeline


def chat(user_input: str, history: list):
    """处理用户对话"""
    pipeline = get_pipeline()

    try:
        result = pipeline.run(user_input, verbose=False)

        # 格式化输出
        intent = result["intent"]
        itinerary = result["itinerary"]
        timing = result["timing"]

        response_parts = [
            f"🧠 **意图解析**（耗时 {timing['intent_parsing']:.1f}s）\n"
            f"```json\n{intent}\n```\n",
            f"🗺️  **行程规划**（耗时 {timing['itinerary_planning']:.1f}s）\n"
            f"{itinerary}\n",
            f"⏱️  总耗时：{timing['total']:.1f}s"
        ]

        return "\n---\n".join(response_parts)

    except Exception as e:
        return f"❌ 处理出错：{str(e)}\n\n请尝试换一种方式描述你的旅行需求。"


# 构建 Gradio 界面
with gr.Blocks(
    title="Travel Agent - Multi-Agent 旅行规划助手",
    theme=gr.themes.Soft()
) as demo:
    gr.Markdown(
        "# 🌍 Travel Agent\n"
        "基于 Multi-Agent + RAG 的智能旅行行程规划系统\n\n"
        "**示例输入**：我想去云南玩5天，情侣出行，喜欢拍照和美食，不想太累"
    )

    chatbot = gr.Chatbot(height=500, label="对话历史")
    msg = gr.Textbox(
        placeholder="输入你的旅行需求...",
        label="你的需求",
        lines=3
    )

    with gr.Row():
        submit_btn = gr.Button("规划行程 🚀", variant="primary")
        clear_btn = gr.Button("清空 🗑️")

    def submit_message(user_input: str, history: list):
        if not user_input.strip():
            return "", history
        bot_response = chat(user_input, history)
        history.append((user_input, bot_response))
        return "", history

    submit_btn.click(fn=submit_message, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(fn=submit_message, inputs=[msg, chatbot], outputs=[msg, chatbot])
    clear_btn.click(fn=lambda: (None, []), outputs=[msg, chatbot])

    gr.Markdown(
        "\n---\n"
        "**技术说明**：三层 Agent 串联 — 意图解析 → 知识检索（RAG）→ 行程编排\n"
        "**Powered by**：LangChain + LlamaIndex + FAISS + BGE-M3 + Ollama"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
