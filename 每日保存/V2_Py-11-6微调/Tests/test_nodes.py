from src.state import GraphState, StepOutput
from src.nodes import reflection_agent_node, RESPONSE_MODEL

# Pytest的测试函数，都以 test_ 开头
def test_reflection_agent_node_replan_scenario():
    """
    Tests the reflection_agent_node under a scenario where it should decide to replan.
    """
    # 1. 准备测试数据 (Arrange)
    #    我们伪造一个history，其中最新的答案与后续计划冲突
    fake_state_for_replan = {
        "original_question": "Lilian Weng的祖父是谁？",
        "plan": ["找出Lilian Weng的父亲是谁", "调查这位父亲的科研生涯"], # 计划是调查科研
        "history": [
            {
                "sub_question": "Lilian Weng的父亲是谁?",
                "retrieved_docs": ["...Alex Weng is an accomplished artist..."],
                "answer": "Lilian Weng的父亲是一位名叫Alex Weng的杰出艺术家。" # 答案是“艺术家”！
            }
        ],
        "messages": [],
        "reflection": {}, # 初始化为空
        "num_cycles": 1
    }

    # 2. 执行被测试的函数 (Act)
    update_result = reflection_agent_node(fake_state_for_replan)

    # 3. 断言结果 (Assert)
    #    我们检查函数的返回值是否符合我们的预期
    assert "reflection" in update_result
    reflection_output = update_result["reflection"]
    
    # 检查决策是否正确
    assert reflection_output["decision"] == "replan"
    
    # 检查是否生成了有意义的反馈
    assert "artist" in reflection_output["feedback_for_planner"] or "画家" in reflection_output["feedback_for_planner"]
    
    print("\n✅ test_reflection_agent_node_replan_scenario PASSED!")

# 你可以为'continue'和'end'场景，编写更多的 test_... 函数
def test_reflection_agent_node_continue_scenario():
    # ... 准备一个应该continue的fake_state ...
    # ... 执行 ...
    # ... 断言 decision == "continue" ...
    pass