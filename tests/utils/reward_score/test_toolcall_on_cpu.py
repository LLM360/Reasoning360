## added by reasoning360

import json
from verl.utils.reward_score.toolcall import compute_score_v0


class TestComputeScoreV0:
    """Unit tests for compute_score_v0 function"""

    def test_correct_solution_with_thinking(self):
        """Test: Correct solution with thinking tags should return 1"""
        ground_truth = json.dumps(
            [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        )
        solution_str = """<|im_start|>assistant
        <think>
        I need to calculate 2+2 which equals 4.
        </think>
        <tool_call>
        [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        </tool_call>"""

        score = compute_score_v0(solution_str, ground_truth)
        assert score == 1

    def test_correct_solution_without_thinking(self):
        """Test: Correct solution without thinking tags should return 0"""
        ground_truth = json.dumps(
            [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        )
        solution_str = """<|im_start|>assistant
        <tool_call>
        [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        </tool_call>"""

        score = compute_score_v0(solution_str, ground_truth)
        assert score == 0

    def test_malformed_json_in_tool_call(self):
        """Test: Malformed JSON in tool call should return 0"""
        ground_truth = json.dumps(
            [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        )
        solution_str = """<|im_start|>assistant
        <think>
        I need to calculate something.
        </think>
        <tool_call>
        [{"name": "calculator", "arguments": {"expression": "2+2"}
        </tool_call>"""

        score = compute_score_v0(solution_str, ground_truth)
        assert score == 0


    def test_wrong_tool_name(self):
        """Test: Wrong tool name should return 0"""
        ground_truth = json.dumps(
            [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        )
        solution_str = """<|im_start|>assistant
        <think>
        I need to calculate 2+2.
        </think>
        <tool_call>
        [{"name": "wrong_tool", "arguments": {"expression": "2+2"}}]
        </tool_call>"""

        score = compute_score_v0(solution_str, ground_truth)
        assert score == 0

    def test_wrong_arguments(self):
        """Test: Correct tool name but wrong arguments should return 0"""
        ground_truth = json.dumps(
            [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        )
        solution_str = """<|im_start|>assistant
        <think>
        I need to calculate 2+3.
        </think>
        <tool_call>
        [{"name": "calculator", "arguments": {"expression": "2+3"}}]
        </tool_call>"""

        score = compute_score_v0(solution_str, ground_truth)
        assert score == 0

    def test_missing_required_fields(self):
        """Test: Missing required fields (name or arguments) should return 0"""
        ground_truth = json.dumps(
            [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        )
        solution_str = """<|im_start|>assistant
        <think>
        I need to calculate 2+2.
        </think>
        <tool_call>
        [{"name": "calculator"}]
        </tool_call>"""

        score = compute_score_v0(solution_str, ground_truth)
        assert score == 0

    def test_multiple_tool_calls(self):
        """Test: Multiple tool calls that match ground truth"""
        ground_truth = json.dumps(
            [
                {"name": "calculator", "arguments": {"expression": "2+2"}},
                {"name": "calculator", "arguments": {"expression": "3*4"}},
            ]
        )
        solution_str = """<|im_start|>assistant
        <think>
        I need to perform two calculations.
        </think>
        <tool_call>
        [
        {"name": "calculator", "arguments": {"expression": "2+2"}},
        {"name": "calculator", "arguments": {"expression": "3*4"}}
        ]
        </tool_call>"""

        score = compute_score_v0(solution_str, ground_truth)
        assert score == 1

    def test_no_tool_call_tag(self):
        """Test: Missing tool_call tags should return 0 (no extraction)"""
        ground_truth = json.dumps(
            [{"name": "calculator", "arguments": {"expression": "2+2"}}]
        )
        solution_str = """<|im_start|>assistant
        <think>
        I need to calculate 2+2.
        </think>
        The result is 4."""

        score = compute_score_v0(solution_str, ground_truth)
        assert score == 0

