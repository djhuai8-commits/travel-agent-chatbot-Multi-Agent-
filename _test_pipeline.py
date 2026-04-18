"""快速测试 pipeline 能否正常调用"""
import os, sys, traceback
os.environ["no_proxy"] = "localhost,127.0.0.1"

sys.path.insert(0, r"C:\Users\27644\WorkBuddy\20260407124956\travel-agent-chatbot")

print("[1/4] 导入模块...", flush=True)
from src.pipeline import TravelAgentPipeline, build_retriever_from_json
from pathlib import Path

data_path = Path(r"C:\Users\27644\WorkBuddy\20260407124956\travel-agent-chatbot\data\sample_travel_guides.json")

retriever = None
if data_path.exists():
    print(f"[2/4] 加载知识库: {data_path.name}", flush=True)
    retriever = build_retriever_from_json(str(data_path))
else:
    print("[2/4] 数据文件不存在，使用无检索模式", flush=True)

print("[3/4] 创建 Pipeline...", flush=True)
p = TravelAgentPipeline(hybrid_retriever=retriever, use_local=True)

print("[4/4] 调用 run() (这步可能较慢，请等待)...", flush=True)
try:
    r = p.run("测试", verbose=False)
    print(f"SUCCESS: keys={list(r.keys())}", flush=True)
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
