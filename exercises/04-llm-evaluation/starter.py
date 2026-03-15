"""
Exercise 4: LLM Evaluation — Testing AI That Thinks
=====================================================
Build a comprehensive evaluation pipeline with golden datasets,
LLM-as-judge, adversarial testing, RAG evaluation, and prompt
regression testing.

Instructions: Fill in each TODO section. Run with: python starter.py
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client, wrappers
from openai import AsyncOpenAI, OpenAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from pydantic import BaseModel, Field
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

# ============================================================
# Environment Setup
# ============================================================

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
openai_client = wrappers.wrap_openai(OpenAI())
ls_client = Client()

GOLDEN_DATASET_NAME = "eval-workshop-golden"
ADVERSARIAL_DATASET_NAME = "eval-workshop-adversarial"


# ============================================================
# STEP 1: Build a Golden Dataset
# ============================================================

# TODO: Create a golden dataset in LangSmith with at least 5 Q&A pairs.
# - Use ls_client.create_dataset() to create the dataset
# - Use ls_client.create_examples() to add examples
# - Each example should have "inputs" (with a "question" key)
#   and "outputs" (with an "answer" key)
#
# Hint: Clean up any existing dataset first for re-runnability:
#   try:
#       existing = ls_client.read_dataset(dataset_name=GOLDEN_DATASET_NAME)
#       ls_client.delete_dataset(dataset_id=existing.id)
#   except Exception:
#       pass
#
# Example structure:
#   examples = [
#       {
#           "inputs": {"question": "What is Python?"},
#           "outputs": {"answer": "Python is a high-level programming language."},
#       },
#       ...
#   ]

def create_golden_dataset() -> None:
    pass  # <-- Replace with your implementation


# ============================================================
# STEP 2: Define the Target Function
# ============================================================

# TODO: Create the function that we will evaluate.
# - Takes inputs: dict with a "question" key
# - Calls the LLM with a system prompt and the user question
# - Returns a dict with an "answer" key
#
# Hint: Use the wrapped OpenAI client for automatic tracing:
#   response = openai_client.chat.completions.create(
#       model="gpt-4o",
#       messages=[
#           {"role": "system", "content": "You are a software engineering tutor..."},
#           {"role": "user", "content": inputs["question"]},
#       ],
#   )
#   return {"answer": response.choices[0].message.content.strip()}

SYSTEM_PROMPT = (
    "You are a senior software engineering tutor. "
    "Answer questions accurately and concisely. "
    "If you don't know something, say so."
)


def target_function(inputs: dict) -> dict:
    pass  # <-- Replace with your implementation


# ============================================================
# STEP 3: Build an LLM-as-Judge Evaluator
# ============================================================

# TODO: Create a correctness evaluator using OpenEvals.
# - Use create_llm_as_judge() with CORRECTNESS_PROMPT
# - Set model to "openai:gpt-4o" and feedback_key to "correctness"
# - Wrap it in a function that takes inputs, outputs, reference_outputs
#
# Hint:
#   judge = create_llm_as_judge(
#       prompt=CORRECTNESS_PROMPT,
#       model="openai:gpt-4o",
#       feedback_key="correctness",
#   )
#   def correctness_evaluator(inputs, outputs, reference_outputs):
#       return judge(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    pass  # <-- Replace with your implementation


# ============================================================
# STEP 4: Build Custom Code Evaluators
# ============================================================

# TODO: Create a length check evaluator.
# - Check that the answer has between 5 and 500 words
# - Return a dict with "key", "score" (0.0 or 1.0), and "comment"
#
# def evaluate_length(inputs: dict, outputs: dict) -> dict:
#     word_count = len(outputs.get("answer", "").split())
#     is_valid = 5 <= word_count <= 500
#     return {"key": "length_check", "score": 1.0 if is_valid else 0.0, "comment": ...}

def evaluate_length(inputs: dict, outputs: dict) -> dict:
    pass  # <-- Replace with your implementation


# TODO: Create a system prompt leak detector.
# - Check that the answer doesn't contain phrases like "system prompt",
#   "you are a", "your instructions"
# - Return a dict with "key", "score", and "comment"

def evaluate_no_system_leak(inputs: dict, outputs: dict) -> dict:
    pass  # <-- Replace with your implementation


# ============================================================
# STEP 5: Build Adversarial Test Cases
# ============================================================

# TODO: Create an adversarial dataset with at least 4 tricky inputs:
# 1. A prompt injection attempt
# 2. A question about something fictional (hallucination trap)
# 3. An out-of-scope question
# 4. A DAN-style jailbreak attempt
#
# Each example should have "inputs" and "outputs" (the expected safe response).
#
# Then create a safety evaluator using a custom LLM judge that checks:
# - The model didn't follow injection instructions
# - The model didn't hallucinate
# - The model declined out-of-scope requests politely

def create_adversarial_dataset() -> None:
    pass  # <-- Replace with your implementation


# TODO: Create an adversarial safety evaluator.
# - Use a structured output model (Pydantic BaseModel with is_safe and reasoning)
# - The judge should check if the response is safe and appropriate
# - Return dict with "key": "adversarial_safety", "score", and "comment"

def evaluate_adversarial_safety(inputs: dict, outputs: dict) -> dict:
    pass  # <-- Replace with your implementation


# ============================================================
# STEP 6: RAG Evaluation with RAGAS
# ============================================================

# TODO: Create SingleTurnSample objects and evaluate them with RAGAS.
# - Create at least 2 samples with user_input, response, reference, and retrieved_contexts
# - One sample should be faithful, one should hallucinate beyond the context
# - Use DiscreteMetric to create a faithfulness evaluator
# - Run the evaluator asynchronously
#
# Hint:
#   faithfulness_metric = DiscreteMetric(
#       name="faithfulness",
#       allowed_values=["faithful", "not_faithful"],
#       prompt="Evaluate if the response is faithful to the retrieved contexts...",
#   )
#   score = await faithfulness_metric.ascore(
#       llm=evaluator_llm,
#       response=sample.response,
#       retrieved_contexts="\n".join(sample.retrieved_contexts),
#   )

async def evaluate_rag_samples() -> None:
    pass  # <-- Replace with your implementation


# ============================================================
# STEP 7: Prompt Regression Testing
# ============================================================

# TODO: Compare two prompt versions on the same dataset.
# - Define a BASELINE_PROMPT and a CANDIDATE_PROMPT
# - Create a factory function that builds a target with a given prompt
# - Run both through ls_client.evaluate() on the golden dataset
# - Use different experiment_prefix values so you can compare them
#
# Hint:
#   def create_target_with_prompt(system_prompt: str):
#       def target(inputs: dict) -> dict:
#           response = openai_client.chat.completions.create(
#               model="gpt-4o",
#               messages=[
#                   {"role": "system", "content": system_prompt},
#                   {"role": "user", "content": inputs["question"]},
#               ],
#           )
#           return {"answer": response.choices[0].message.content.strip()}
#       return target

BASELINE_PROMPT = (
    "You are a senior software engineering tutor. "
    "Answer questions accurately and concisely. "
    "If you don't know something, say so."
)

CANDIDATE_PROMPT = (
    "You are a friendly coding mentor. "
    "Answer questions in simple terms, using analogies where helpful. "
    "Keep answers brief. If unsure, say you're not sure."
)


def run_prompt_regression() -> None:
    pass  # <-- Replace with your implementation


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("Layer 4 Workshop — Exercise 4: LLM Evaluation")
    print("=" * 60)

    # Step 1: Create golden dataset
    print("\n--- Step 1: Creating golden dataset ---")
    create_golden_dataset()

    # Step 2 & 3: Evaluate with LLM-as-judge
    print("\n--- Steps 2-3: Running golden dataset evaluation ---")
    if target_function({"question": "test"}) is None:
        print("ERROR: target_function not implemented. Complete TODO in Step 2.")
    else:
        results = ls_client.evaluate(
            target_function,
            data=GOLDEN_DATASET_NAME,
            evaluators=[
                correctness_evaluator,
                evaluate_length,
                evaluate_no_system_leak,
            ],
            experiment_prefix="workshop-golden-eval",
            max_concurrency=2,
        )
        print("Golden evaluation complete!")

    # Step 5: Adversarial evaluation
    print("\n--- Step 5: Running adversarial evaluation ---")
    create_adversarial_dataset()
    if target_function({"question": "test"}) is not None:
        adversarial_results = ls_client.evaluate(
            target_function,
            data=ADVERSARIAL_DATASET_NAME,
            evaluators=[
                evaluate_adversarial_safety,
                evaluate_no_system_leak,
                evaluate_length,
            ],
            experiment_prefix="workshop-adversarial-eval",
            max_concurrency=2,
        )
        print("Adversarial evaluation complete!")

    # Step 6: RAG evaluation
    print("\n--- Step 6: RAG evaluation with RAGAS ---")
    asyncio.run(evaluate_rag_samples())

    # Step 7: Prompt regression
    print("\n--- Step 7: Prompt regression testing ---")
    run_prompt_regression()

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 60)
    print("\nView results at: https://smith.langchain.com")
