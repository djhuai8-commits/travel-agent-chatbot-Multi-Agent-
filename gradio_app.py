# ============================================================
# 旅行行程规划系统 - Gradio 可视化界面（携程风格）
# ============================================================

import os
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

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

        intent = result["intent"]
        itinerary = result["itinerary"]
        timing = result["timing"]

        # 格式化输出为 HTML 卡片
        intent_json = f"""<pre style="background:#f5f5f5;padding:12px;border-radius:8px;overflow-x:auto;font-size:13px;">{intent}</pre>"""

        response_html = f"""
        <div class="response-card">
            <div class="card-header">
                <span class="card-icon">🧠</span>
                <span class="card-title">意图解析</span>
                <span class="card-time">{timing['intent_parsing']:.1f}s</span>
            </div>
            <div class="card-body">{intent_json}</div>
        </div>

        <div class="response-card highlight">
            <div class="card-header">
                <span class="card-icon">🗺️</span>
                <span class="card-title">为你定制的行程</span>
                <span class="card-time">{timing['itinerary_planning']:.1f}s</span>
            </div>
            <div class="card-body itinerary-body">{itinerary}</div>
        </div>

        <div class="total-time">⏱️ 总耗时：{timing['total']:.1f}s</div>
        """

        return response_html

    except Exception as e:
        return f"""
        <div class="error-card">
            <span>❌ 处理出错：{str(e)}</span>
            <p>请尝试换一种方式描述你的旅行需求。</p>
        </div>
        """


# ============================================================
# 携程风格 CSS
# ============================================================

CTRIP_CSS = """
/* ---- 顶部导航栏 ---- */
.nav-bar {
    background: linear-gradient(135deg, #0066CC 0%, #004999 100%);
    padding: 0 40px;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 0;
    margin-bottom: 0;
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}

.nav-logo-text {
    color: #ffffff;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.nav-logo-icon {
    font-size: 24px;
}

.nav-links {
    display: flex;
    gap: 28px;
    align-items: center;
}

.nav-link {
    color: rgba(255,255,255,0.85);
    font-size: 14px;
    cursor: pointer;
    transition: color 0.2s;
    text-decoration: none;
}

.nav-link:hover, .nav-link.active {
    color: #ffffff;
}

.nav-link-cta {
    background: #FF6600;
    color: white !important;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
}

/* ---- Hero 搜索区 ---- */
.hero-section {
    background: linear-gradient(160deg, #e8f4fd 0%, #dbeeff 50%, #f0f7ff 100%);
    padding: 40px 40px 30px;
    border-bottom: 1px solid #e0e0e0;
}

.hero-title {
    color: #003366;
    font-size: 28px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 6px;
}

.hero-subtitle {
    color: #666;
    font-size: 14px;
    text-align: center;
    margin-bottom: 24px;
}

.search-container {
    max-width: 720px;
    margin: 0 auto;
    position: relative;
}

.search-box {
    display: flex;
    background: white;
    border-radius: 28px;
    box-shadow: 0 4px 20px rgba(0,102,204,0.15);
    overflow: hidden;
    border: 2px solid #0066CC;
}

.search-input-wrap {
    flex: 1;
    padding: 4px 4px 4px 20px;
    display: flex;
    align-items: center;
}

.search-input-wrap input {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    font-size: 15px;
    width: 100%;
    padding: 10px 0;
    color: #333;
    background: transparent !important;
}

.search-input-wrap input::placeholder {
    color: #aaa;
}

/* ---- 真实输入框（Hero 搜索区位置） ---- */
.hero-search-input {
    max-width: 720px;
    margin: 0 auto 8px !important;
}
.hero-search-input textarea {
    width: 100% !important;
    border: 2px solid #0066CC !important;
    border-radius: 28px !important;
    padding: 16px 24px !important;
    font-size: 15px !important;
    outline: none !important;
    box-shadow: 0 4px 20px rgba(0,102,204,0.15) !important;
    background: white !important;
    resize: none !important;
    color: #333 !important;
}
.hero-search-input textarea:focus {
    border-color: #004999 !important;
    box-shadow: 0 6px 24px rgba(0,102,204,0.22) !important;
}

.search-btn button {
    background: linear-gradient(135deg, #FF6600, #ff8533) !important;
    color: white !important; border: none !important;
    border-radius: 26px !important; padding: 14px 32px !important;
    font-size: 16px !important; font-weight: 700 !important;
    cursor: pointer; transition: all 0.2s;
    letter-spacing: 1px;
}
.search-btn button:hover {
    background: linear-gradient(135deg, #e65500, #ff701a) !important;
}

/* ---- 对话区 ---- */
.chat-section {
    max-width: 860px;
    margin: 0 auto;
    padding: 30px 40px;
    min-height: 400px;
}

.chat-example-hint {
    text-align: center;
    color: #999;
    font-size: 13px;
    margin-bottom: 20px;
}

.chat-example-hint span {
    background: #f0f7ff;
    color: #0066CC;
    padding: 4px 12px;
    border-radius: 14px;
    margin: 0 4px;
    cursor: pointer;
    border: 1px solid #cce4ff;
    font-size: 12px;
}

/* Chatbot 气泡 */
.gradio-container .chatbot {
    border: none !important;
    background: transparent !important;
}

.chat-message {
    display: flex;
    margin-bottom: 20px;
    align-items: flex-start;
}

.chat-message.user {
    flex-direction: row-reverse;
}

.chat-bubble-avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
    margin: 0 10px;
}

.chat-bubble-avatar.user-avatar {
    background: linear-gradient(135deg, #0066CC, #004999);
}

.chat-bubble-avatar.bot-avatar {
    background: linear-gradient(135deg, #FF6600, #ff8533);
}

.bubble-content {
    max-width: 72%;
}

.user .bubble-content {
    text-align: right;
}

.user-bubble {
    background: linear-gradient(135deg, #0066CC 0%, #0055aa 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    font-size: 14px;
    line-height: 1.6;
    display: inline-block;
    max-width: 100%;
    word-break: break-word;
}

.bot-bubble {
    background: white;
    border: 1px solid #e8e8e8;
    border-radius: 4px 18px 18px 18px;
    padding: 16px 20px;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    max-width: 100%;
    word-break: break-word;
}

/* 响应卡片样式 */
.response-card {
    background: white;
    border: 1px solid #eee;
    border-radius: 12px;
    margin-bottom: 16px;
    overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

.response-card.highlight {
    border: 1px solid #cce4ff;
    box-shadow: 0 2px 16px rgba(0,102,204,0.1);
}

.card-header {
    background: #f8f9fa;
    padding: 10px 16px;
    display: flex;
    align-items: center;
    gap: 8px;
    border-bottom: 1px solid #eee;
}

.card-icon {
    font-size: 16px;
}

.card-title {
    font-weight: 600;
    color: #333;
    font-size: 14px;
    flex: 1;
}

.card-time {
    color: #999;
    font-size: 12px;
}

.card-body {
    padding: 14px 16px;
    color: #444;
    font-size: 14px;
    line-height: 1.7;
}

.itinerary-body {
    white-space: pre-wrap;
}

.total-time {
    text-align: center;
    color: #888;
    font-size: 12px;
    padding: 8px;
    margin-top: 4px;
}

.error-card {
    background: #fff5f5;
    border: 1px solid #ffcccc;
    border-radius: 12px;
    padding: 16px 20px;
    color: #c00;
    font-size: 14px;
}

.error-card p {
    color: #666;
    font-size: 13px;
    margin-top: 8px;
}

/* ---- 底部 Footer ---- */
.footer {
    text-align: center;
    padding: 20px;
    color: #bbb;
    font-size: 12px;
    border-top: 1px solid #eee;
    margin-top: 20px;
}

/* ---- Gradio 覆盖 ---- */
#app-header { display: none !important; }
footer { display: none !important; }

.gradio-container {
    font-family: "PingFang SC", "Microsoft YaHei", "Helvetica Neue", sans-serif !important;
}

button.primary {
    background: linear-gradient(135deg, #FF6600, #ff8533) !important;
    border: none !important;
}

/* 空状态 */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #aaa;
}

.empty-state-icon {
    font-size: 48px;
    margin-bottom: 12px;
}

.empty-state-text {
    font-size: 15px;
}

/* 热词标签 */
.hot-tags {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 16px;
    flex-wrap: wrap;
}

.hot-tag {
    background: white;
    border: 1px solid #ddeeff;
    color: #0066CC;
    padding: 5px 14px;
    border-radius: 16px;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s;
}

.hot-tag:hover {
    background: #0066CC;
    color: white;
}
"""


# ============================================================
# 构建 Gradio 界面
# ============================================================

with gr.Blocks(
    title="TripGenie 智能行程规划",
    css=CTRIP_CSS,
    head='<meta charset="UTF-8">'
) as demo:

    # ---- 顶部导航 ----
    gr.HTML("""
    <div class="nav-bar">
        <div class="nav-logo">
            <span class="nav-logo-icon">🌍</span>
            <span class="nav-logo-text">TripGenie</span>
        </div>
        <div class="nav-links">
            <a class="nav-link active">首页</a>
            <a class="nav-link">目的地</a>
            <a class="nav-link">旅行灵感</a>
            <a class="nav-link">我的行程</a>
            <a class="nav-link nav-link-cta">立即规划</a>
        </div>
    </div>
    """)

    # ---- Hero 区 + 输入框（合并为真实控件） ----
    gr.HTML("""
    <div class="hero-section">
        <div class="hero-title">智能旅行规划助手</div>
        <div class="hero-subtitle">基于 Multi-Agent + RAG 深度理解你的需求，打造专属行程</div>
    </div>
    """)

    msg = gr.Textbox(
        placeholder="描述你的旅行梦想，比如：我想去云南玩5天，情侣出行，喜欢拍照和美食...",
        label="",
        lines=2,
        elem_classes=["hero-search-input"]
    )

    # 热词提示
    gr.HTML("""
    <div style="max-width:720px;margin:0 auto 12px;padding:0 40px;display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
        <span class="hot-tag" onclick="document.querySelector('textarea').value='去厦门3天亲子游推荐'">🌸 厦门亲子游</span>
        <span class="hot-tag" onclick="document.querySelector('textarea').value='重庆美食4天攻略'">🍜 重庆美食</span>
        <span class="hot-tag" onclick="document.querySelector('textarea').value='情侣去大理怎么安排'">🏔️ 大理浪漫游</span>
        <span class="hot-tag" onclick="document.querySelector('textarea').value='带老人去成都三天怎么安排'">🧘 成都休闲</span>
    </div>
    """)

    # 按钮行（紧跟输入框）
    with gr.Row():
        submit_btn = gr.Button("🚀 开始规划行程", elem_classes="search-btn")
        clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", size="sm")

    # ---- 对话区域 ----
    gr.HTML("""
    <div class="chat-section" style="max-width:860px;margin:0 auto;padding:30px 40px;min-height:400px;">
    </div>
    """)

    chatbot = gr.Chatbot(
        height=520,
        show_label=False,
        container=True,
        elem_classes=["chat-bubble-container"]
    )

    # 热词提示（放在对话区下方）
    gr.HTML("""
    <div style="max-width:720px;margin:12px auto 0;padding:0 40px;display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
        <span style="font-size:12px;color:#bbb;">试试这样问：</span>
        <span class="hot-tag">去厦门3天亲子游推荐</span>
        <span class="hot-tag">重庆美食4天攻略</span>
        <span class="hot-tag">情侣去大理怎么安排</span>
    </div>
    """)

    # Footer
    gr.HTML("""
    <div class="footer">
        <p>Powered by LangChain · LlamaIndex · Ollama · RAG 知识检索</p>
        <p>© 2025 TripGenie · Multi-Agent 旅行规划系统</p>
    </div>
    """)

    # ---- 交互逻辑 ----
    def submit_message(user_input: str, history: list):
        if not user_input.strip():
            return "", history

        # 添加用户消息（messages 格式，Gradio 6.x 默认要求）
        history.append({"role": "user", "content": user_input})

        # 生成回复
        bot_response = chat(user_input, history)

        # 添加助手回复
        history.append({"role": "assistant", "content": bot_response})

        return "", history

    def click_suggestion(text: str, history: list):
        """点击推荐词时自动填充并提交"""
        return submit_message(text, history)

    submit_btn.click(
        fn=submit_message,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot]
    )

    # 清空
    clear_btn.click(
        fn=lambda: (None, []),
        outputs=[msg, chatbot]
    )

    # 页面加载时聚焦输入框
    demo.load(
        fn=None,
        js="""() => { setTimeout(() => { var t = document.querySelector('textarea'); if (t) t.focus(); }, 800); }"""
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
