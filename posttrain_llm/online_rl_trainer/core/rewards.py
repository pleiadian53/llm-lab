"""
Reward functions for Online RL training.

Provides both verifiable rewards (math, code) and composite rewards.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


class RewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    @abstractmethod
    def __call__(
        self, 
        completions: List[List[Dict[str, str]]], 
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            completions: List of completions, each completion is a list of messages
                        with 'role' and 'content' keys
            **kwargs: Additional arguments (e.g., ground_truth, test_cases)
            
        Returns:
            List of reward values (one per completion)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this reward function."""
        pass


class MathRewardFunction(RewardFunction):
    """
    Reward function for math problems with verifiable answers.
    
    Extracts answer from \\boxed{} format and compares to ground truth.
    """
    
    def __init__(
        self, 
        normalize_numbers: bool = True,
        case_sensitive: bool = False
    ):
        """
        Args:
            normalize_numbers: Whether to normalize numeric formats (e.g., "72.0" == "72")
            case_sensitive: Whether string comparison is case-sensitive
        """
        self.normalize_numbers = normalize_numbers
        self.case_sensitive = case_sensitive
        self._pattern = re.compile(r"\\boxed\{(.*?)\}")
    
    @property
    def name(self) -> str:
        return "math_boxed"
    
    def _extract_answer(self, content: str) -> str:
        """Extract answer from \\boxed{} format."""
        match = self._pattern.search(content)
        if match:
            return match.group(1).strip()
        return ""
    
    def _normalize(self, value: str) -> str:
        """Normalize answer for comparison."""
        value = value.strip()
        
        if not self.case_sensitive:
            value = value.lower()
        
        if self.normalize_numbers:
            # Remove commas from numbers
            value = value.replace(",", "")
            # Try to normalize numeric values
            try:
                num = float(value)
                # Convert to int if it's a whole number
                if num == int(num):
                    return str(int(num))
                return str(num)
            except ValueError:
                pass
        
        return value
    
    def __call__(
        self, 
        completions: List[List[Dict[str, str]]], 
        ground_truth: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards by comparing extracted answers to ground truth.
        
        Args:
            completions: Model completions
            ground_truth: List of correct answers
            
        Returns:
            List of rewards (1.0 for correct, 0.0 for incorrect)
        """
        rewards = []
        for completion, gt in zip(completions, ground_truth):
            content = completion[0].get("content", "")
            extracted = self._extract_answer(content)
            
            # Normalize both for comparison
            extracted_norm = self._normalize(extracted)
            gt_norm = self._normalize(gt)
            
            reward = 1.0 if extracted_norm == gt_norm else 0.0
            rewards.append(reward)
        
        return rewards


class CodeRewardFunction(RewardFunction):
    """
    Reward function for code problems with test cases.
    
    Executes generated code against test cases in a sandbox.
    """
    
    def __init__(
        self, 
        timeout: float = 5.0,
        partial_credit: bool = True
    ):
        """
        Args:
            timeout: Execution timeout in seconds
            partial_credit: Whether to give partial credit for passing some tests
        """
        self.timeout = timeout
        self.partial_credit = partial_credit
        self._code_pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    
    @property
    def name(self) -> str:
        return "code_execution"
    
    def _extract_code(self, content: str) -> str:
        """Extract Python code from markdown code blocks."""
        match = self._code_pattern.search(content)
        if match:
            return match.group(1).strip()
        # Fallback: try to find any code-like content
        return content.strip()
    
    def _run_test(self, code: str, test_input: str, expected_output: str) -> bool:
        """Run a single test case (simplified - use sandbox in production)."""
        try:
            # WARNING: This is simplified. Use a proper sandbox in production!
            local_vars = {}
            exec(code, {"__builtins__": {}}, local_vars)
            
            # Find the main function
            func = None
            for name, obj in local_vars.items():
                if callable(obj):
                    func = obj
                    break
            
            if func is None:
                return False
            
            result = func(test_input)
            return str(result) == expected_output
            
        except Exception:
            return False
    
    def __call__(
        self, 
        completions: List[List[Dict[str, str]]], 
        test_cases: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards by running code against test cases.
        
        Args:
            completions: Model completions containing code
            test_cases: List of test cases, each is a list of dicts with
                       'input' and 'expected_output' keys
            
        Returns:
            List of rewards (0.0 to 1.0 based on tests passed)
        """
        rewards = []
        for completion, tests in zip(completions, test_cases):
            content = completion[0].get("content", "")
            code = self._extract_code(content)
            
            if not tests:
                rewards.append(0.0)
                continue
            
            passed = sum(
                self._run_test(code, tc["input"], tc["expected_output"])
                for tc in tests
            )
            
            if self.partial_credit:
                reward = passed / len(tests)
            else:
                reward = 1.0 if passed == len(tests) else 0.0
            
            rewards.append(reward)
        
        return rewards


class FormatRewardFunction(RewardFunction):
    """
    Reward function for format compliance.
    
    Checks if response follows required format (e.g., has thinking tags).
    """
    
    def __init__(
        self,
        required_patterns: Optional[List[str]] = None,
        forbidden_patterns: Optional[List[str]] = None
    ):
        """
        Args:
            required_patterns: Regex patterns that must be present
            forbidden_patterns: Regex patterns that must not be present
        """
        self.required_patterns = [
            re.compile(p) for p in (required_patterns or [])
        ]
        self.forbidden_patterns = [
            re.compile(p) for p in (forbidden_patterns or [])
        ]
    
    @property
    def name(self) -> str:
        return "format_compliance"
    
    def __call__(
        self, 
        completions: List[List[Dict[str, str]]], 
        **kwargs
    ) -> List[float]:
        """
        Compute rewards based on format compliance.
        
        Returns:
            List of rewards (1.0 if all requirements met, 0.0 otherwise)
        """
        rewards = []
        for completion in completions:
            content = completion[0].get("content", "")
            
            # Check required patterns
            all_required = all(
                p.search(content) for p in self.required_patterns
            )
            
            # Check forbidden patterns
            no_forbidden = not any(
                p.search(content) for p in self.forbidden_patterns
            )
            
            reward = 1.0 if (all_required and no_forbidden) else 0.0
            rewards.append(reward)
        
        return rewards


@dataclass
class RewardWeight:
    """Weight configuration for a reward function in composite rewards."""
    reward_fn: RewardFunction
    weight: float = 1.0


class CompositeRewardFunction(RewardFunction):
    """
    Combines multiple reward functions with weights.
    
    Final reward = sum(weight_i * reward_i) / sum(weight_i)
    """
    
    def __init__(self, reward_weights: List[RewardWeight]):
        """
        Args:
            reward_weights: List of RewardWeight objects
        """
        self.reward_weights = reward_weights
        self._total_weight = sum(rw.weight for rw in reward_weights)
    
    @property
    def name(self) -> str:
        names = [rw.reward_fn.name for rw in self.reward_weights]
        return f"composite({'+'.join(names)})"
    
    def __call__(
        self, 
        completions: List[List[Dict[str, str]]], 
        **kwargs
    ) -> List[float]:
        """
        Compute weighted combination of rewards.
        
        Returns:
            List of weighted average rewards
        """
        batch_size = len(completions)
        combined_rewards = [0.0] * batch_size
        
        for rw in self.reward_weights:
            rewards = rw.reward_fn(completions, **kwargs)
            for i, r in enumerate(rewards):
                combined_rewards[i] += rw.weight * r
        
        # Normalize by total weight
        return [r / self._total_weight for r in combined_rewards]


# Convenience factory functions
def create_math_reward(
    normalize_numbers: bool = True,
    require_boxed: bool = True
) -> RewardFunction:
    """Create a math reward function."""
    if require_boxed:
        return MathRewardFunction(normalize_numbers=normalize_numbers)
    else:
        # Could add alternative extraction methods here
        return MathRewardFunction(normalize_numbers=normalize_numbers)


def create_format_reward(
    require_thinking_tags: bool = False,
    require_boxed: bool = True
) -> FormatRewardFunction:
    """Create a format reward function."""
    required = []
    if require_thinking_tags:
        required.append(r"<think>.*</think>")
    if require_boxed:
        required.append(r"\\boxed\{.*\}")
    
    return FormatRewardFunction(required_patterns=required)
