# 检索后端质量对比：以评测集 cases/*.json 已有的 knowledge_query + knowledge_hits_contain
# 标注作为金标准，逐 query 报告期望命中在 BM25 / TF-IDF 两个后端中的排名位置，
# 并汇总 MRR 与平均排名。只读、无副作用、输出确定（同语料同标注结果可复现）。
# 用法：python evaluation/compare_retrievers.py
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge.retriever import BM25Retriever, TFIDFRetriever  # noqa: E402

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases")


def collect_gold_queries():
    """从评测用例提取 (case_id, query, 期望命中标题) 三元组，按文件名与轮次顺序。"""
    queries = []
    for filename in sorted(os.listdir(CASES_DIR)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(CASES_DIR, filename), encoding="utf-8") as fh:
            case = json.load(fh)
        for turn in case.get("turns", []):
            expect = turn.get("expect") or {}
            if "knowledge_query" not in expect:
                continue
            for wanted in expect.get("knowledge_hits_contain", []):
                queries.append((case.get("id", filename), expect["knowledge_query"], wanted))
    return queries


def rank_of(retriever, query, wanted, top_k=20):
    """返回期望标题在检索结果中的 1-based 排名，未命中返回 None。"""
    for position, hit in enumerate(retriever.search(query, top_k=top_k), start=1):
        if hit["title"] == wanted:
            return position
    return None


def summarize(ranks):
    """由 1-based 排名列表返回 (MRR, 平均排名)；空列表返回 (0.0, 0.0)。"""
    if not ranks:
        return 0.0, 0.0
    reciprocal = sum(1.0 / rank for rank in ranks)
    return reciprocal / len(ranks), sum(ranks) / len(ranks)


def main():
    gold = collect_gold_queries()
    if not gold:
        print("评测用例中未找到 knowledge_query 标注，无法对比")
        return 1

    backends = {"BM25": BM25Retriever(), "TF-IDF": TFIDFRetriever()}
    ranks_by_backend = {name: [] for name in backends}

    print(f"{'case':<30} {'query':<10} {'期望命中':<18} " + " ".join(f"{name:>8}" for name in backends))
    for case_id, query, wanted in gold:
        row = [f"{case_id:<30}", f"{query:<10}", f"{wanted:<18}"]
        for name, retriever in backends.items():
            position = rank_of(retriever, query, wanted)
            if position is not None:
                ranks_by_backend[name].append(position)
                row.append(f"{position:>8}")
            else:
                row.append(f"{'miss':>8}")
        print(" ".join(row))

    print("-" * 72)
    for name in backends:
        mrr, avg_rank = summarize(ranks_by_backend[name])
        print(f"{name}: 命中 {len(ranks_by_backend[name])}/{len(gold)}，MRR={mrr:.3f}，平均排名={avg_rank:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
