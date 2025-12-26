
import sys
import os
import logging

# プロジェクトルートをパスに追加
sys.path.append(os.getcwd())

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("grace_integration_test")

from grace.planner import create_planner
from grace.executor import create_executor
from grace.schemas import ExecutionPlan

def test_grace_native_integration():
    logger.info("--- Starting GRACE Native Integration Test ---")
    
    # 1. PlannerとExecutorの準備
    planner = create_planner()
    executor = create_executor()
    
    # 2. 質問の定義
    # wikipedia_jaにはなさそうで、livedoorにありそうなトピック（アプリレビューなど）
    query = "iPhoneのTODOリストアプリ「Clear」の特徴は何ですか？"
    logger.info(f"User Query: {query}")
    
    # 3. 計画生成
    plan = planner.create_plan(query)
    logger.info(f"Generated Plan: {len(plan.steps)} steps")
    for s in plan.steps:
        logger.info(f"  - Step {s.step_id}: {s.action} ({s.description})")

    # 4. 計画実行
    logger.info("Executing plan...")
    result = executor.execute_plan(plan)
    
    # 5. 結果の表示
    print("\n" + "="*50)
    print(f"FINAL STATUS: {result.overall_status}")
    print(f"OVERALL CONFIDENCE: {result.overall_confidence:.2f}")
    print(f"FINAL ANSWER:\n{result.final_answer}")
    print("="*50)
    
    if result.overall_status in ["success", "partial"] and result.final_answer:
        print("\nSUCCESS: GRACE Native execution returned a response.")
        if "有田幸樹" not in result.final_answer:
             print("SUCCESS: Irrelevant keywords (Arita Koki) were filtered out or ignored.")
        else:
             print("WARNING: Irrelevant keywords found in answer.")
    else:
        print("\nFAILED: Execution did not complete as expected.")

if __name__ == "__main__":
    test_grace_native_integration()
