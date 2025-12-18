# eval/evaluator.py
"""
エージェント評価フレームワーク
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable # Added Tuple, Callable
from dataclasses import dataclass
import logging

# Configure logging for evaluation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """テストケース定義"""
    id: str
    category: str
    input: str
    expected_tool_use: bool
    subcategory: str = ""
    expected_collection: Optional[str] = None
    expected_tool_name: Optional[str] = None
    expected_behavior: Optional[str] = None
    description: str = ""

@dataclass
class TestResult:
    """テスト結果"""
    test_case_id: str
    category: str
    input: str
    actual_tool_used: bool
    actual_tool_name: Optional[str] = None
    actual_collection: Optional[str] = None
    response: str = ""
    latency_ms: float = 0.0
    top_score: float = 0.0
    passed: bool = False
    failure_reason: str = ""
    timestamp: str = ""

def load_test_cases(path: str = "eval/test_cases.json") -> List[TestCase]:
    """テストケースをJSONから読み込み"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return [TestCase(**tc) for tc in data["test_cases"]]
    except FileNotFoundError:
        logger.error(f"Test case file not found at: {path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {path}: {e}")
        return []

def evaluate_routing(test_case: TestCase, actual_tool_used: bool) -> Tuple[bool, str]:
    """ルーティング判断の評価"""
    if test_case.expected_tool_use == actual_tool_used:
        return True, ""
    else:
        return False, f"Expected tool_use={test_case.expected_tool_use}, got {actual_tool_used}"

def evaluate_collection_selection(
    test_case: TestCase,
    actual_collection: Optional[str]
) -> Tuple[bool, str]:
    """コレクション選択の評価"""
    if test_case.expected_collection is None:
        return True, ""
    # None check for actual_collection when expectation is set
    if actual_collection is None:
        return False, f"Expected collection='{test_case.expected_collection}', but no collection was selected."
        
    if test_case.expected_collection == actual_collection:
        return True, ""
    return False, f"Expected collection='{test_case.expected_collection}', got '{actual_collection}'"

def evaluate_tool_name(
    test_case: TestCase,
    actual_tool_name: Optional[str]
) -> Tuple[bool, str]:
    """ツール名の評価"""
    if test_case.expected_tool_name is None:
        return True, ""
    if actual_tool_name is None:
        return False, f"Expected tool='{test_case.expected_tool_name}', but no tool was used."
        
    if test_case.expected_tool_name == actual_tool_name:
        return True, ""
    return False, f"Expected tool='{test_case.expected_tool_name}', got '{actual_tool_name}'"

def evaluate_hallucination(
    test_case: TestCase,
    response: str,
    search_results_found: bool
) -> Tuple[bool, str]:
    """ハルシネーション評価 (No Result時の挙動チェック)"""
    if test_case.expected_behavior != "not_found_response":
        return True, ""

    not_found_keywords: List[str] = [
        "見つかりませんでした", "情報がありません", "該当する結果がありません",
        "関連性スコアが低い", "回答が見つかりませんでした",
        "情報源の中には", "情報が見つかりませんでした"
    ]

    for keyword in not_found_keywords:
        if keyword in response:
            return True, ""
            
    return False, "Expected 'not found' response, but got potential hallucination."

def run_single_test(
    test_case: TestCase,
    agent_func: Callable[[str], Tuple[str, Dict[str, Any]]], # agent_func expects (response_text, tool_info_dict)
    get_metrics_func: Optional[Callable[[], List[Any]]] = None # Optional Callable that returns a list of metrics
) -> TestResult:
    """単一テストの実行"""
    start_time: float = time.time()
    timestamp: str = time.strftime("%Y-%m-%d %H:%M:%S")

    result: TestResult = TestResult(
        test_case_id=test_case.id,
        category=test_case.category,
        input=test_case.input,
        actual_tool_used=False,
        timestamp=timestamp
    )

    try:
        response, tool_info = agent_func(test_case.input) # type: ignore

        result.response = response
        result.actual_tool_used = tool_info.get("tool_used", False)
        result.actual_tool_name = tool_info.get("tool_name")
        result.actual_collection = tool_info.get("collection_name")
        result.latency_ms = (time.time() - start_time) * 1000.0

        if get_metrics_func:
            metrics: List[Any] = get_metrics_func()
            if metrics:
                latest = metrics[-1]
                result.top_score = getattr(latest, 'top_score', 0.0) # Use getattr for safety with Any type

        failures: List[str] = []

        passed, reason = evaluate_routing(test_case, result.actual_tool_used)
        if not passed:
            failures.append(reason)

        if test_case.expected_tool_use and result.actual_tool_used:
            passed, reason = evaluate_tool_name(test_case, result.actual_tool_name)
            if not passed:
                failures.append(reason)

        if result.actual_tool_name == "search_rag_knowledge_base":
            passed, reason = evaluate_collection_selection(test_case, result.actual_collection)
            if not passed:
                failures.append(reason)

        search_found: bool = result.top_score > 0
        passed, reason = evaluate_hallucination(test_case, response, search_found)
        if not passed:
            failures.append(reason)

        result.passed = len(failures) == 0
        result.failure_reason = "; ".join(failures)

    except Exception as e:
        result.failure_reason = f"Exception during execution: {str(e)}"
        result.passed = False
        logger.error(f"Test {test_case.id} failed with exception: {e}", exc_info=True)

    return result

def generate_report(results: List[TestResult]) -> Dict[str, Any]:
    """評価レポート生成"""
    total: int = len(results)
    passed: int = sum(1 for r in results if r.passed)

    by_category: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat: str = r.category
        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0, "failed_ids": []}
        by_category[cat]["total"] += 1
        if r.passed:
            by_category[cat]["passed"] += 1
        else:
            by_category[cat]["failed_ids"].append(r.test_case_id)

    for cat in by_category:
        by_category[cat]["accuracy"] = (
            by_category[cat]["passed"] / by_category[cat]["total"]
            if by_category[cat]["total"] > 0 else 0.0
        )

    latencies: List[float] = [r.latency_ms for r in results if r.latency_ms > 0]
    avg_latency: float = sum(latencies) / len(latencies) if latencies else 0.0
    max_latency: float = max(latencies) if latencies else 0.0
    min_latency: float = min(latencies) if latencies else 0.0


    return {
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "accuracy": passed / total if total > 0 else 0.0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "by_category": by_category,
        "performance": {
            "avg_latency_ms": round(avg_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "min_latency_ms": round(min_latency, 2)
        },
        "failed_cases": [
            {
                "id": r.test_case_id,
                "input": r.input,
                "reason": r.failure_reason,
                "agent_response": r.response
            }
            for r in results if not r.passed
        ]
    }

def save_report(report: Dict[str, Any], output_path: str = "eval/results/report.json") -> None:
    """レポートをJSONファイルに保存"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {output_path}")

def print_report_summary(report: Dict[str, Any]) -> None:
    """レポートサマリーをコンソール出力"""
    print("\n" + "="*60)
    print("📊 評価レポート サマリー")
    print("="*60)

    s: Dict[str, Any] = report["summary"]
    print(f"\n【全体結果】")
    print(f"  総テストケース: {s['total_cases']}")
    print(f"  成功: {s['passed']} / 失敗: {s['failed']}")
    print(f"  精度: {s['accuracy']*100:.1f}%")

    print(f"\n【カテゴリ別】")
    for cat, data in report["by_category"].items():
        status: str = "✅" if data["accuracy"] >= 0.9 else "⚠️" if data["accuracy"] >= 0.7 else "❌"
        print(f"  {status} {cat}: {data['accuracy']*100:.1f}% ({data['passed']}/{data['total']})")

    print(f"\n【パフォーマンス】")
    p: Dict[str, Any] = report["performance"]
    print(f"  平均レイテンシ: {p['avg_latency_ms']}ms")
    print(f"  最大: {p['max_latency_ms']}ms / 最小: {p['min_latency_ms']}ms")

    if report["failed_cases"]:
        print(f"\n【失敗ケース詳細】")
        for fc in report["failed_cases"]:
            print(f"  ❌ {fc['id']}: {fc['reason']} (Response: {fc['agent_response']})") # Agent response for debugging

    print("\n" + "="*60)