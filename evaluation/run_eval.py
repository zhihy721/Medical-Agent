# -*- coding: utf-8 -*-
"""系统化评测集执行脚本。

回放 evaluation/cases/*.json 中的多轮对话剧本，走真实 LangGraph 链路
（默认 mock provider，保证离线与确定性），逐轮对比断言并输出指标。

用法：
    python evaluation/run_eval.py            # 跑全部用例（mock，确定性基线）
    python evaluation/run_eval.py --case 01_high_risk_chest_pain
    python evaluation/run_eval.py --verbose  # 打印每轮详细信息
    python evaluation/run_eval.py --provider deepseek  # 真实 LLM 手动模式（非确定，仅供参考）

用例 JSON 结构：
    {
      "id": "01_xxx",
      "description": "...",
      "turns": [
        {
          "user": "用户话术",
          "pulse_data": {"pulse_summary": "浮紧"},   // 可选：本轮前注入脉诊数据
          "expect": {
            "risk": "HIGH",                          // 期望风险等级
            "next_action": "risk_escalation",        // 期望动作（精确）
            "next_action_any": ["a", "b"],           // 期望动作（任一）
            "is_final": true,                        // 是否最终回复
            "symptoms_contain": ["胸痛"],            // case_state.symptoms 包含
            "red_flags_contain": ["呼吸困难"],       // case_state.red_flags 包含
            "syndromes_contain": ["风寒束表"],       // plan.syndrome_candidates 包含
            "slots_filled": ["cold_heat"],           // 槽位已填充
            "response_contain": ["就医"],            // 回复文本包含
            "knowledge_query": "桂枝汤",             // 直接调知识检索（可选）
            "knowledge_hits_contain": ["桂枝汤"]     // 检索命中名称包含（可选）
          }
        }
      ]
    }
"""
import argparse
import io
import json
import os
import sys
import time

# 保证从任意目录运行时都能导入项目模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.factory import create_agent  # noqa: E402
from llm.llm import LLM  # noqa: E402
from llm.prompt import SYSTEM_PROMPT  # noqa: E402
from memory.memory import ConversationMemory  # noqa: E402
from tools.knowledge_tool import search_knowledge  # noqa: E402

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")


def load_cases(case_filter=None):
    """读取 cases 目录下所有用例，按 id 排序；支持按 id 过滤。"""
    cases = []
    for filename in sorted(os.listdir(CASES_DIR)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(CASES_DIR, filename), "r", encoding="utf-8") as handle:
            case = json.load(handle)
        if case_filter and case.get("id") != case_filter:
            continue
        cases.append(case)
    return cases


def build_agent(case_id, provider="mock"):
    """为单个用例构建隔离的 agent：默认 mock 保证确定性；provider 可透传真实后端。"""
    llm = LLM(system_prompt=SYSTEM_PROMPT, provider=provider)
    memory = ConversationMemory(session_id=f"eval-{case_id}", user_id="eval-user")
    return create_agent(llm, memory, runtime="langgraph")


def run_case(case, provider="mock"):
    """回放一个用例的所有轮次，返回 (逐轮记录列表, 运行错误)。"""
    agent = build_agent(case["id"], provider=provider)
    records = []
    error = ""
    try:
        for turn in case.get("turns", []):
            if turn.get("pulse_data"):
                agent.ingest_pulse_data(turn["pulse_data"])
            response = agent.run(turn["user"])
            values = agent.get_graph_state().values
            records.append(
                {
                    "user": turn["user"],
                    "response": response,
                    "plan": values.get("plan") or {},
                    "risk_result": values.get("risk_result") or {},
                    "action_result": values.get("action_result") or {},
                    "case_state": agent.memory.get_case_state(),
                }
            )
    except Exception as exc:  # 记录错误但不中断整体评测
        error = f"{type(exc).__name__}: {exc}"
    return records, error


def check_turn(expect, record):
    """按断言逐项检查一轮结果，返回 (失败原因列表, 红旗 tp, fp, fn)。"""
    failures = []
    flag_tp = flag_fp = flag_fn = 0
    plan = record["plan"]
    risk_result = record["risk_result"]
    action_result = record["action_result"]
    case_state = record["case_state"]

    if "risk" in expect and risk_result.get("risk") != expect["risk"]:
        failures.append(f"risk 期望 {expect['risk']}，实际 {risk_result.get('risk')}")

    if "next_action" in expect and plan.get("next_action") != expect["next_action"]:
        failures.append(f"next_action 期望 {expect['next_action']}，实际 {plan.get('next_action')}")

    if "next_action_any" in expect and plan.get("next_action") not in expect["next_action_any"]:
        failures.append(f"next_action 期望 {expect['next_action_any']} 之一，实际 {plan.get('next_action')}")

    if "is_final" in expect and bool(action_result.get("is_final")) != bool(expect["is_final"]):
        failures.append(f"is_final 期望 {expect['is_final']}，实际 {action_result.get('is_final')}")

    for symptom in expect.get("symptoms_contain", []):
        if symptom not in case_state.get("symptoms", []):
            failures.append(f"symptoms 未包含 {symptom}，实际 {case_state.get('symptoms', [])}")

    for flag in expect.get("red_flags_contain", []):
        if flag not in case_state.get("red_flags", []):
            failures.append(f"red_flags 未包含 {flag}，实际 {case_state.get('red_flags', [])}")

    # 红旗精确集合标注：既算断言，也累计 precision/recall 的 TP/FP/FN
    if "red_flags_set" in expect:
        expected_flags = set(expect["red_flags_set"])
        actual_flags = set(case_state.get("red_flags", []))
        flag_tp = len(expected_flags & actual_flags)
        flag_fp = len(actual_flags - expected_flags)
        flag_fn = len(expected_flags - actual_flags)
        if actual_flags != expected_flags:
            failures.append(f"red_flags 集合期望 {sorted(expected_flags)}，实际 {sorted(actual_flags)}")

    candidate_names = [item.get("name") for item in plan.get("syndrome_candidates", [])]
    for name in expect.get("syndromes_contain", []):
        if name not in candidate_names:
            failures.append(f"syndrome_candidates 未包含 {name}，实际 {candidate_names}")

    for slot in expect.get("slots_filled", []):
        if not case_state.get(slot):
            failures.append(f"槽位 {slot} 未填充，实际值 {case_state.get(slot)!r}")

    if expect.get("contradictions_empty") and case_state.get("contradictions"):
        failures.append(f"contradictions 应为空，实际 {case_state.get('contradictions')}")

    for fragment in expect.get("response_contain", []):
        if fragment not in record["response"]:
            failures.append(f"response 未包含 {fragment!r}")

    # 知识检索断言：直接调 search_knowledge 校验命中，不依赖主流程的调用时机；
    # 窗口取 8（大于默认 5），避免新增相关条目把原有期望项挤出小窗口造成误报
    if "knowledge_query" in expect:
        retrieval = search_knowledge(expect["knowledge_query"], top_k=8)
        hit_names = [hit["name"] for hit in retrieval["hits"]]
        for wanted in expect.get("knowledge_hits_contain", []):
            if wanted not in hit_names:
                failures.append(f"knowledge 命中未包含 {wanted}，实际 {hit_names}")

    return failures, flag_tp, flag_fp, flag_fn


def evaluate_case(case, verbose=False, provider="mock"):
    """执行并评估一个用例，返回结果 dict。"""
    records, error = run_case(case, provider=provider)
    turns = case.get("turns", [])
    turn_results = []
    all_failures = []
    red_flag_tp = red_flag_fp = red_flag_fn = 0

    for index, turn in enumerate(turns):
        expect = turn.get("expect") or {}
        if index >= len(records):
            turn_results.append({"pass": False, "failures": [f"缺少第 {index + 1} 轮运行记录"]})
            continue
        if expect:
            failures, flag_tp, flag_fp, flag_fn = check_turn(expect, records[index])
        else:
            failures, flag_tp, flag_fp, flag_fn = [], 0, 0, 0
        red_flag_tp += flag_tp
        red_flag_fp += flag_fp
        red_flag_fn += flag_fn
        turn_results.append({"pass": not failures, "failures": failures})
        all_failures.extend(f"[轮 {index + 1}] {message}" for message in failures)

    if error:
        all_failures.append(f"[运行错误] {error}")

    # 汇总指标：风险断言命中、收敛轮次、replan 情况
    risk_expected = 0
    risk_correct = 0
    for index, turn in enumerate(turns):
        expect = turn.get("expect") or {}
        if "risk" not in expect or index >= len(records):
            continue
        risk_expected += 1
        if records[index]["risk_result"].get("risk") == expect["risk"]:
            risk_correct += 1

    convergence_turn = None
    for index, record in enumerate(records):
        if record["plan"].get("next_action") == "final_advice":
            convergence_turn = index + 1
            break

    replanned_turns = sum(
        1 for record in records if "自检改判" in (record["plan"].get("action_reason") or "")
    )

    return {
        "id": case.get("id"),
        "description": case.get("description", ""),
        "pass": not all_failures,
        "failures": all_failures,
        "turn_results": turn_results,
        "risk_expected": risk_expected,
        "risk_correct": risk_correct,
        "turn_count": len(turns),
        "convergence_turn": convergence_turn,
        "replanned_turns": replanned_turns,
        "red_flag_tp": red_flag_tp,
        "red_flag_fp": red_flag_fp,
        "red_flag_fn": red_flag_fn,
    }


def aggregate_metrics(results):
    """从用例结果汇总全局指标，供控制台报告与 report.json 共用。"""
    passed = sum(1 for item in results if item["pass"])
    total = len(results)
    risk_expected = sum(item["risk_expected"] for item in results)
    risk_correct = sum(item["risk_correct"] for item in results)
    converged = [item for item in results if item["convergence_turn"] is not None]
    total_replans = sum(item["replanned_turns"] for item in results)
    total_turns = sum(item["turn_count"] for item in results)

    flag_tp = sum(item["red_flag_tp"] for item in results)
    flag_fp = sum(item["red_flag_fp"] for item in results)
    flag_fn = sum(item["red_flag_fn"] for item in results)
    flag_precision = round(flag_tp / (flag_tp + flag_fp), 3) if flag_tp + flag_fp else None
    flag_recall = round(flag_tp / (flag_tp + flag_fn), 3) if flag_tp + flag_fn else None

    return {
        "cases_passed": passed,
        "cases_total": total,
        "risk_expected": risk_expected,
        "risk_correct": risk_correct,
        "risk_accuracy": round(risk_correct / risk_expected, 3) if risk_expected else None,
        "red_flag_tp": flag_tp,
        "red_flag_fp": flag_fp,
        "red_flag_fn": flag_fn,
        "red_flag_precision": flag_precision,
        "red_flag_recall": flag_recall,
        "converged_cases": len(converged),
        "avg_convergence_turns": round(sum(item["convergence_turn"] for item in converged) / len(converged), 2)
        if converged
        else None,
        "replanned_turns": total_replans,
        "total_turns": total_turns,
        "replan_rate": round(total_replans / total_turns, 3) if total_turns else None,
    }


def build_report(results):
    """构造可落盘的评测报告：时间戳 + 全局指标 + 逐用例结果。"""
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "metrics": aggregate_metrics(results),
        "cases": [
            {
                "id": item["id"],
                "pass": item["pass"],
                "failures": item["failures"],
                "convergence_turn": item["convergence_turn"],
                "replanned_turns": item["replanned_turns"],
            }
            for item in results
        ],
    }


def print_report(results, verbose=False):
    """打印评测报告：明细 + 汇总指标。"""
    metrics = aggregate_metrics(results)

    print("=" * 60)
    print(f"评测结果：{metrics['cases_passed']}/{metrics['cases_total']} 用例通过")
    print("=" * 60)

    for item in results:
        status = "PASS" if item["pass"] else "FAIL"
        convergence = f"，收敛于第 {item['convergence_turn']} 轮" if item["convergence_turn"] else ""
        print(f"[{status}] {item['id']} - {item['description']}{convergence}")
        if not item["pass"]:
            for failure in item["failures"]:
                print(f"    x {failure}")
        if verbose:
            for index, turn in enumerate(item["turn_results"]):
                print(f"    轮 {index + 1}: {'通过' if turn['pass'] else '失败'}")

    print("-" * 60)
    if metrics["risk_expected"]:
        print(
            f"风险识别准确率: {metrics['risk_correct']}/{metrics['risk_expected']} "
            f"({round(metrics['risk_correct'] / metrics['risk_expected'] * 100, 1)}%)"
        )
    if metrics["red_flag_tp"] + metrics["red_flag_fp"] + metrics["red_flag_fn"]:
        print(
            f"红旗识别 precision: {metrics['red_flag_precision']}，recall: {metrics['red_flag_recall']} "
            f"(TP={metrics['red_flag_tp']} FP={metrics['red_flag_fp']} FN={metrics['red_flag_fn']})"
        )
    if metrics["converged_cases"]:
        print(f"收敛用例数: {metrics['converged_cases']}，平均收敛轮数: {metrics['avg_convergence_turns']}")
    if metrics["total_turns"]:
        print(
            f"replan 轮数占比: {metrics['replanned_turns']}/{metrics['total_turns']} "
            f"({round(metrics['replanned_turns'] / metrics['total_turns'] * 100, 1)}%)"
        )
    print("-" * 60)


def main(argv=None):
    # Windows 控制台输出统一 UTF-8，避免中文乱码
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in {"utf-8", "utf8"}:
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="Medical-Agent 系统化评测集")
    parser.add_argument("--case", help="只运行指定 id 的用例")
    parser.add_argument("--verbose", action="store_true", help="打印每轮详细结果")
    parser.add_argument(
        "--provider",
        default="mock",
        help="LLM provider（默认 mock 保证确定性；真实后端输出非确定，失败仅供参考）",
    )
    parser.add_argument("--report", metavar="PATH", help="将评测报告 JSON 落盘到指定路径")
    args = parser.parse_args(argv)

    if args.provider != "mock":
        print(f"[提示] 使用真实 provider={args.provider}：输出非确定，失败仅供参考，不计入 CI 基线")

    cases = load_cases(args.case)
    if not cases:
        print(f"未找到用例（--case {args.case}），请检查 evaluation/cases/")
        return 1

    results = [evaluate_case(case, verbose=args.verbose, provider=args.provider) for case in cases]
    print_report(results, verbose=args.verbose)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(build_report(results), handle, ensure_ascii=False, indent=2)
        print(f"评测报告已写入 {args.report}")

    return 0 if all(item["pass"] for item in results) else 1


if __name__ == "__main__":
    sys.exit(main())
