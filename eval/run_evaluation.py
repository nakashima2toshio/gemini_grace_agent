# eval/run_evaluation.py
"""
評価実行スクリプト
Usage: python -m eval.run_evaluation
"""
import sys
import logging
from pathlib import Path
import os
from typing import List, Dict, Any, Callable, Tuple # Added Callable, Tuple
from google.generativeai import ChatSession # Corrected import

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluator import (
    load_test_cases,
    run_single_test,
    generate_report,
    save_report,
    print_report_summary,
    TestCase, # Import TestCase for type hinting
    TestResult # Import TestResult for type hinting
)
from agent_main import setup_agent, run_agent_turn
from agent_tools import get_search_metrics, clear_search_metrics, SearchMetrics # Import SearchMetrics for type hinting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger: logging.Logger = logging.getLogger(__name__)


def agent_wrapper(user_input: str) -> Tuple[str, Dict[str, Any]]:
    """
    Evaluator用のエージェントラッパー関数
    """
    chat: ChatSession = setup_agent()
    # run_agent_turn は (response_text, tool_info_dict) を返す
    response_text: str
    tool_info: Dict[str, Any]
    response_text, tool_info = run_agent_turn(chat, user_input, return_tool_info=True) # type: ignore
    return response_text, tool_info


def main() -> None:
    """評価メイン処理"""
    logger.info("評価プロセスを開始します...")

    # テストケース読み込み
    test_cases_path: str = "eval/test_cases.json"
    logger.info(f"テストケースを読み込み中: {test_cases_path}")
    test_cases: List[TestCase] = load_test_cases(test_cases_path)
    
    if not test_cases:
        logger.error("有効なテストケースが見つかりません。終了します。")
        sys.exit(1)
        
    logger.info(f"ロードされたテストケース数: {len(test_cases)}")

    # メトリクスクリア
    clear_search_metrics()

    # テスト実行
    results: List[TestResult] = []
    total: int = len(test_cases)
    
    for i, tc in enumerate(test_cases, 1):
        logger.info(f"[{i}/{total}] Running Test: {tc.id} - {tc.description}")
        logger.info(f"  Input: {tc.input}")
        
        result: TestResult = run_single_test(tc, agent_wrapper, get_search_metrics) # type: ignore
        results.append(result)

        status: str = "✅ PASS" if result.passed else "❌ FAIL"
        logger.info(f"  Result: {status}")
        if not result.passed:
            logger.warning(f"  Reason: {result.failure_reason}")
            logger.info(f"  Response: {result.response[:100]}...")

    # レポート生成
    report: Dict[str, Any] = generate_report(results)

    # 保存
    save_report(report)
    
    # サマリー出力
    print_report_summary(report)

    # 終了コード（CI用）
    if report["summary"]["accuracy"] < 0.9:
        logger.warning("精度が目標値(90%)を下回っています。")
        # sys.exit(1) # CI環境では有効化する

    logger.info("評価プロセス完了")


if __name__ == "__main__":
    main()