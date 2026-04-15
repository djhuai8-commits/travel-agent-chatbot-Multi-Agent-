# ============================================================
# 旅行行程规划系统 - 评测脚本
# ============================================================

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from src.pipeline import TravelAgentPipeline, build_retriever_from_json


@dataclass
class EvalCase:
    """评测用例"""
    user_input: str
    expected_destinations: List[str]
    expected_days: Dict[str, int]
    tolerance_days: int = 1
    notes: str = ""


# ============================================================
# 评测数据集（覆盖三类场景）
# ============================================================

EVAL_DATASET: List[EvalCase] = [
    # --- 简单单城 ---
    EvalCase(
        user_input="去成都玩3天，喜欢吃辣",
        expected_destinations=["成都"],
        expected_days={"成都": 3},
        notes="单城简单查询"
    ),
    EvalCase(
        user_input="我想去厦门看海，情侣出行，5天左右",
        expected_destinations=["厦门"],
        expected_days={"厦门": 5},
        notes="单城+明确偏好"
    ),
    EvalCase(
        user_input="带60岁爸妈去北京玩，不爬长城",
        expected_destinations=["北京"],
        expected_days={"北京": 3},
        notes="老年出行，注意避免高强度"
    ),
    # --- 跨城多日 ---
    EvalCase(
        user_input="上海出发，厦门3天然后福州1天，情侣出行",
        expected_destinations=["厦门", "福州"],
        expected_days={"厦门": 3, "福州": 1},
        notes="跨城多日，检验顺序逻辑"
    ),
    EvalCase(
        user_input="我想去云南玩7天，丽江2天，大理2天，香格里拉3天",
        expected_destinations=["丽江", "大理", "香格里拉"],
        expected_days={"丽江": 2, "大理": 2, "香格里拉": 3},
        notes="三城串行，涉及交通规划"
    ),
    EvalCase(
        user_input="去西北玩8天，西宁2天，青海湖2天，敦煌3天，张掖1天",
        expected_destinations=["西宁", "青海湖", "敦煌", "张掖"],
        expected_days={"西宁": 2, "青海湖": 2, "敦煌": 3, "张掖": 1},
        notes="多城长途环线"
    ),
    # --- 含特殊偏好 ---
    EvalCase(
        user_input="带孩子去三亚，5天，喜欢挖沙和游泳，不想安排太累",
        expected_destinations=["三亚"],
        expected_days={"三亚": 5},
        notes="亲子出行，轻松节奏"
    ),
    EvalCase(
        user_input="摄影爱好者去喀纳斯，秋天，想拍晨雾和金色白桦林",
        expected_destinations=["喀纳斯"],
        expected_days={"喀纳斯": 4},
        notes="小众目的地，摄影专项"
    ),
    EvalCase(
        user_input="清真饮食习惯，去西安5天，和朋友一起",
        expected_destinations=["西安"],
        expected_days={"西安": 5},
        notes="饮食限制，需检索清真餐厅"
    ),
    EvalCase(
        user_input="我想去恩施大峡谷玩3天，情侣出行，不爬山",
        expected_destinations=["恩施"],
        expected_days={"恩施": 3},
        notes="小众目的地冷查询测试"
    ),
]


# ============================================================
# 评估函数
# ============================================================

def evaluate_case(pipeline: TravelAgentPipeline, case: EvalCase) -> Dict[str, Any]:
    """对单个用例进行评估"""
    result = pipeline.run(case.user_input, verbose=False)
    intent = result["intent"]
    itinerary = result["itinerary"]

    metrics = {
        "dest_precision": len(
            set(intent.get("destinations", [])) & set(case.expected_destinations)
        ) / max(len(intent.get("destinations", [])), 1),
        "dest_recall": len(
            set(intent.get("destinations", [])) & set(case.expected_destinations)
        ) / max(len(case.expected_destinations), 1),
        "days_mentioned_correctly": _check_days(itinerary, case.expected_days),
        "preference_match": _check_preference_match(itinerary, intent),
        "logic_correctness": _check_logic_errors(itinerary),
        "response_time": result["timing"]["total"],
        "output_chars": len(itinerary)
    }

    return {
        "case": case,
        "intent": intent,
        "itinerary_preview": itinerary[:200] + "..." if len(itinerary) > 200 else itinerary,
        "metrics": metrics,
        "timing": result["timing"]
    }


def _check_days(itinerary: str, expected_days: Dict[str, int]) -> float:
    total_expected = sum(expected_days.values())
    day_mentions = len(re.findall(r'Day ?\d|第 ?[一二三四五六七八九十\d]+ ?天', itinerary))
    if total_expected == 0:
        return 0.0
    return min(day_mentions / total_expected, 1.0)


def _check_preference_match(itinerary: str, intent: dict) -> float:
    dislikes = intent.get("dislikes", [])
    score = 1.0
    for dislike in dislikes:
        if dislike in itinerary:
            score -= 0.2
    return max(score, 0.0)


def _check_logic_errors(itinerary: str) -> float:
    days = re.split(r'Day ?\d|第 ?[一二三四五六七八九十\d]+ ?天', itinerary)
    errors = 0
    for day_content in days:
        spots = [s.strip() for s in day_content.split('：') if s.strip()]
        if len(spots) > 5:
            errors += 1
    max_errors = max(len(days) - 1, 1)
    return max(1 - errors / max_errors, 0.0)


# ============================================================
# 运行完整评测
# ============================================================

def run_evaluation(
    retriever=None,
    output_path: str = "eval_results.json",
    verbose: bool = True
) -> Dict[str, Any]:
    pipeline = TravelAgentPipeline(hybrid_retriever=retriever, use_local=True)
    results = []
    total_time = 0.0

    for i, case in enumerate(EVAL_DATASET):
        if verbose:
            print(f"[{i+1}/{len(EVAL_DATASET)}] {case.user_input[:30]}...")
        eval_result = evaluate_case(pipeline, case)
        results.append(eval_result)
        total_time += eval_result["timing"]["total"]
        if verbose:
            m = eval_result["metrics"]
            print(f"   目的地召回: {m['dest_recall']:.0%} | "
                  f"逻辑正确性: {m['logic_correctness']:.0%} | "
                  f"耗时: {m['response_time']:.1f}s")

    summary = {
        "total_cases": len(results),
        "total_time_seconds": total_time,
        "avg_time_seconds": total_time / len(results),
        "metrics_avg": {
            "avg_dest_precision": sum(r["metrics"]["dest_precision"] for r in results) / len(results),
            "avg_dest_recall": sum(r["metrics"]["dest_recall"] for r in results) / len(results),
            "avg_days_mentioned": sum(r["metrics"]["days_mentioned_correctly"] for r in results) / len(results),
            "avg_preference_match": sum(r["metrics"]["preference_match"] for r in results) / len(results),
            "avg_logic_correctness": sum(r["metrics"]["logic_correctness"] for r in results) / len(results),
            "avg_response_time": sum(r["metrics"]["response_time"] for r in results) / len(results),
        }
    }

    if verbose:
        print("\n" + "=" * 60)
        print("📊 评测汇总")
        print("=" * 60)
        for k, v in summary["metrics_avg"].items():
            suffix = "s" if "time" in k else ""
            print(f"  {k}: {v:.1%}{suffix}" if suffix else f"  {k}: {v:.2f}s")

    output = {
        "summary": summary,
        "per_case_results": [
            {"user_input": r["case"].user_input, "intent": r["intent"],
             "metrics": r["metrics"], "timing": r["timing"]}
            for r in results
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 结果已保存至 {output_path}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()
    run_evaluation(output_path=args.output)
