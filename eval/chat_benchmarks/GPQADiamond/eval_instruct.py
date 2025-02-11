import json
import logging
import random
from typing import Any, Dict, List, Optional
from datasets import load_dataset

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark

from .testing_utils import get_multiple_choice_answer

import lm_eval.models
from lm_eval.models.vllm_causallms import VLLM


PROMPT = """Return your final response within \\boxed{{}} and only include the letter choice (A, B, C, or D) as your final response.
Problem: {problem}
Options: {options}
Answer:"""


class GPQADiamondBenchmark(BaseBenchmark):
    """
    GPQADiamond Benchmark for evaluating multiple choice reasoning of LLMs.
    Link: https://huggingface.co/datasets/Idavidrein/gpqa
    """

    def __init__(
        self,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_prompt: Optional[str] = None,
        temperature: float = None,
        top_p: float = None,
    ):
        """
        Initialize GPQADiamond benchmark.

        Args:
            debug: If set, only evaluate on 2 examples
            logger: Optional logger instance
        """
        super().__init__(logger)
        self.dataset_name = "Idavidrein/gpqa"
        self.debug = debug
        self.max_new_tokens = 32768
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and examples
        """
        examples = self.load_questions()

        # Prepare instances for model
        all_instances = []

        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        elif isinstance(model, lm_eval.models.openai_completions.LocalCompletionsAPI):
            model_name = model.model
        else:
            model_name = model.model_args["model"]

        # Create the system-formatted message.
        system_fmt_message = {"role": "system", "content": self.system_prompt}

        for idx, example in enumerate(examples):
            multiple_choice_string, correct_answer = self.generate_multiple_choice_answers(example)
            example["answer"] = correct_answer

            messages = [
                system_fmt_message,
                {"role": "user", "content": PROMPT.format(problem=example["Question"], options=multiple_choice_string)},
            ]

            generation_args = {
                "do_sample": True,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }

            templated_messages = model.apply_chat_template(messages)

            all_instances.append(
                Instance(
                    "generate_until",
                    example,
                    (templated_messages, generation_args),
                    idx,
                )
            )

        # Generate model responses
        self.logger.info("Generating responses for GPQADiamond...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, output in zip(examples, outputs):
            example["model_output"] = output
            example["model_answer"] = get_multiple_choice_answer(output)

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""
        if results is None:
            return None

        examples = results["examples"]
        total = len(examples)
        solved = sum(example["answer"] == example["model_answer"] for example in examples)

        results.update(
            {
                "num_total": total,
                "num_solved": solved,
                "accuracy": solved / total,
            }
        )

        return results

    def load_questions(self) -> List[Dict[str, Any]]:
        """Load GPQADiamond questions from the dataset."""
        dataset = load_dataset(self.dataset_name, "gpqa_diamond")
        questions = [row for row in dataset["train"]]
        if self.debug:
            questions = questions[:2]
        self.logger.info(f"Loaded {len(questions)} questions from {self.dataset_name}")
        return questions

    def generate_multiple_choice_answers(self, data: Dict[str, Any]) -> tuple[str, str]:
        """Generate multiple choice string and correct answer letter."""
        answers = [
            data["Correct Answer"],
            data["Incorrect Answer 1"],
            data["Incorrect Answer 2"],
            data["Incorrect Answer 3"],
        ]
        random.shuffle(answers)

        options = ["A", "B", "C", "D"]
        options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

        multiple_choice_string = ", ".join(f"{letter}) {options_to_answers[letter]}" for letter in options)
        correct_answer_letter = next(
            letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"]
        )

        return multiple_choice_string, correct_answer_letter
