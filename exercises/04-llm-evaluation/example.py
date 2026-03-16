"""
DEMO: LLM Evaluation Building Blocks
======================================
Run this BEFORE starting the exercise. It builds up from simple code-based
evaluators to LLM-as-judge patterns and RAG evaluation, so you can see how
each evaluation concept works in isolation.

Run with: python example.py
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

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
openai_client = wrappers.wrap_openai(OpenAI())
ls_client = Client()


# ─────────────────────────────────────────────────────────────
def demo_1_why_exact_match_fails():
    """Show that LLM outputs are non-deterministic."""
    print("=" * 60)
    print("DEMO 1: Why Exact Match Testing Fails")
    print("=" * 60)

    creative_llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    question = "What is Python?"
    responses = []
    for i in range(3):
        response = creative_llm.invoke([
            ("system", "You are a helpful assistant. Keep answers under 2 sentences."),
            ("user", question),
        ])
        responses.append(response.content)
        print(f"  Run {i+1}: {response.content}")

    unique_count = len(set(responses))
    print(f"\n  Unique responses: {unique_count}/3")
    print("  All answers are CORRECT but NOT IDENTICAL — assert == would fail.")
    print()


# ─────────────────────────────────────────────────────────────
def demo_2_code_evaluators():
    """Simple code-based evaluators — fast and deterministic."""
    print("=" * 60)
    print("DEMO 2: Code-Based Evaluators")
    print("=" * 60)

    def evaluate_length(outputs: dict) -> dict:
        answer = outputs.get("answer", "")
        word_count = len(answer.split())
        is_valid = 5 <= word_count <= 500
        return {"key": "length_check", "score": 1.0 if is_valid else 0.0, "words": word_count}

    def evaluate_no_system_leak(outputs: dict) -> dict:
        answer = outputs.get("answer", "").lower()
        leak_indicators = ["system prompt", "you are a", "your instructions"]
        leaked = any(indicator in answer for indicator in leak_indicators)
        return {"key": "no_system_leak", "score": 0.0 if leaked else 1.0}

    test_cases = [
        {"answer": "Python is a high-level programming language known for readability."},
        {"answer": "Yes."},
        {"answer": "As per my system prompt, you are a helpful assistant."},
    ]

    for i, outputs in enumerate(test_cases):
        print(f"\n  Case {i+1}: \"{outputs['answer'][:50]}...\"")
        length_result = evaluate_length(outputs)
        leak_result = evaluate_no_system_leak(outputs)
        print(f"    Length: {'PASS' if length_result['score'] else 'FAIL'} ({length_result['words']} words)")
        print(f"    Leak:   {'PASS' if leak_result['score'] else 'FAIL'}")
    print()


# ─────────────────────────────────────────────────────────────
def demo_3_llm_as_judge():
    """LLM-as-judge with OpenEvals prebuilt evaluator."""
    print("=" * 60)
    print("DEMO 3: LLM-as-Judge (OpenEvals)")
    print("=" * 60)

    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:gpt-4o",
        feedback_key="correctness",
    )

    cases = [
        {
            "label": "Correct answer",
            "inputs": {"question": "What is the capital of France?"},
            "outputs": {"answer": "The capital of France is Paris."},
            "reference_outputs": {"answer": "Paris is the capital of France."},
        },
        {
            "label": "Incorrect answer",
            "inputs": {"question": "What is the capital of France?"},
            "outputs": {"answer": "The capital of France is London."},
            "reference_outputs": {"answer": "Paris is the capital of France."},
        },
    ]

    for case in cases:
        result = evaluator(
            inputs=case["inputs"],
            outputs=case["outputs"],
            reference_outputs=case["reference_outputs"],
        )
        print(f"\n  {case['label']}:")
        print(f"    Score: {result['score']}")
        print(f"    Comment: {str(result.get('comment', 'N/A'))[:100]}")
    print()


# ─────────────────────────────────────────────────────────────
def demo_4_custom_judge():
    """Custom LLM-as-judge with structured output."""
    print("=" * 60)
    print("DEMO 4: Custom LLM-as-Judge (Structured Output)")
    print("=" * 60)

    class HelpfulnessScore(BaseModel):
        score: int = Field(description="Score from 1-5", ge=1, le=5)
        reasoning: str = Field(description="Brief explanation")

    structured_judge = llm.with_structured_output(HelpfulnessScore)

    cases = [
        {
            "question": "How do I reverse a list in Python?",
            "answer": "Use list.reverse() for in-place, list[::-1] for a copy, or reversed(list) for an iterator.",
        },
        {
            "question": "How do I reverse a list in Python?",
            "answer": "Use a loop.",
        },
    ]

    for case in cases:
        result = structured_judge.invoke([
            (
                "system",
                "Score the helpfulness of this answer 1-5.\n"
                "1=Not helpful, 2=Slightly, 3=Moderate, 4=Very, 5=Exceptional",
            ),
            ("user", f"Question: {case['question']}\nAnswer: {case['answer']}"),
        ])
        print(f"\n  Q: {case['question']}")
        print(f"  A: {case['answer']}")
        print(f"  Score: {result.score}/5 — {result.reasoning}")
    print()


# ─────────────────────────────────────────────────────────────
def demo_5_langsmith_evaluation():
    """End-to-end evaluation with LangSmith datasets and experiments."""
    print("=" * 60)
    print("DEMO 5: LangSmith Evaluation Pipeline")
    print("=" * 60)

    dataset_name = "eval-demo-golden"
    try:
        existing = ls_client.read_dataset(dataset_name=dataset_name)
        ls_client.delete_dataset(dataset_id=existing.id)
    except Exception:
        pass

    dataset = ls_client.create_dataset(
        dataset_name=dataset_name,
        description="Demo golden dataset for LLM evaluation",
    )

    examples = [
        {
            "inputs": {"question": "What is Python?"},
            "outputs": {"answer": "Python is a high-level, interpreted programming language."},
        },
        {
            "inputs": {"question": "What is O(n log n)?"},
            "outputs": {"answer": "O(n log n) is a time complexity common in efficient sorting algorithms."},
        },
        {
            "inputs": {"question": "What is a REST API?"},
            "outputs": {"answer": "A REST API uses HTTP methods to perform CRUD operations on resources."},
        },
    ]
    ls_client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"  Created dataset '{dataset_name}' with {len(examples)} examples")

    def target(inputs: dict) -> dict:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer accurately and concisely."},
                {"role": "user", "content": inputs["question"]},
            ],
        )
        return {"answer": response.choices[0].message.content.strip()}

    correctness_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:gpt-4o",
        feedback_key="correctness",
    )

    def correctness_eval(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        return correctness_judge(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)

    print("  Running evaluation...")
    results = ls_client.evaluate(
        target,
        data=dataset_name,
        evaluators=[correctness_eval],
        experiment_prefix="demo-eval",
        max_concurrency=2,
    )

    print("  Evaluation complete! View results at: https://smith.langchain.com")
    print()


# ─────────────────────────────────────────────────────────────
def demo_6_rag_evaluation():
    """RAG-specific evaluation with RAGAS metrics."""
    print("=" * 60)
    print("DEMO 6: RAG Evaluation with RAGAS")
    print("=" * 60)

    async_client = AsyncOpenAI()
    evaluator_llm = llm_factory("gpt-4o", client=async_client)

    faithfulness_metric = DiscreteMetric(
        name="faithfulness",
        allowed_values=["faithful", "not_faithful"],
        prompt=(
            "Evaluate if the response is faithful to the retrieved contexts. "
            "A faithful response only contains claims supported by the given contexts.\n\n"
            "Retrieved Contexts:\n{retrieved_contexts}\n\n"
            "Response: {response}\n\n"
            "Answer with only 'faithful' or 'not_faithful'."
        ),
    )

    samples = [
        SingleTurnSample(
            user_input="What is LangGraph?",
            response="LangGraph provides a StateGraph class for building workflows. It supports conditional edges, loops, and checkpointing.",
            reference="LangGraph is a framework for building stateful, multi-step AI workflows.",
            retrieved_contexts=[
                "LangGraph provides a StateGraph class for building workflows.",
                "LangGraph supports conditional edges, loops, and checkpointing.",
            ],
        ),
        SingleTurnSample(
            user_input="What is the capital of Mars?",
            response="The capital of Mars is Olympus City, founded in 2045.",
            reference="Mars does not have a capital city.",
            retrieved_contexts=[
                "Mars is the fourth planet from the Sun.",
                "Mars has been explored by NASA rovers.",
            ],
        ),
    ]

    async def run_evaluation() -> None:
        for i, sample in enumerate(samples):
            print(f"\n  Sample {i+1}: {sample.user_input}")
            print(f"  Response: {sample.response}")
            score = await faithfulness_metric.ascore(
                llm=evaluator_llm,
                response=sample.response,
                retrieved_contexts="\n".join(sample.retrieved_contexts),
            )
            status = "PASS" if score.value == "faithful" else "FAIL"
            print(f"  Faithfulness: [{status}] {score.value}")

    asyncio.run(run_evaluation())
    print()


# ─────────────────────────────────────────────────────────────
def demo_7_prompt_regression():
    """Compare two prompt versions on the same dataset."""
    print("=" * 60)
    print("DEMO 7: Prompt Regression Testing")
    print("=" * 60)

    dataset_name = "eval-demo-golden"

    baseline_prompt = (
        "You are a senior software engineering tutor. "
        "Answer questions accurately and concisely. "
        "If you don't know something, say so."
    )
    candidate_prompt = (
        "You are a friendly coding mentor. "
        "Answer questions in simple terms, using analogies where helpful. "
        "Keep answers brief. If unsure, say you're not sure."
    )

    def make_target(system_prompt: str):
        def target(inputs: dict) -> dict:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": inputs["question"]},
                ],
            )
            return {"answer": response.choices[0].message.content.strip()}
        return target

    correctness_judge = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model="openai:gpt-4o",
        feedback_key="correctness",
    )

    def correctness_eval(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        return correctness_judge(inputs=inputs, outputs=outputs, reference_outputs=reference_outputs)

    print("  Running baseline prompt evaluation...")
    ls_client.evaluate(
        make_target(baseline_prompt),
        data=dataset_name,
        evaluators=[correctness_eval],
        experiment_prefix="demo-regression-baseline",
        max_concurrency=2,
    )
    print("  Baseline complete!")

    print("  Running candidate prompt evaluation...")
    ls_client.evaluate(
        make_target(candidate_prompt),
        data=dataset_name,
        evaluators=[correctness_eval],
        experiment_prefix="demo-regression-candidate",
        max_concurrency=2,
    )
    print("  Candidate complete!")
    print()
    print("  Compare in LangSmith: open the dataset, select both experiments, click 'Compare'.")
    print("  Look for: per-example score drops, average score changes, new failures.")
    print()


if __name__ == "__main__":
    print()
    print("Layer 4 Workshop — LLM Evaluation Demo")
    print("This demo walks through each evaluation concept one by one.")
    print()

    demo_1_why_exact_match_fails()
    demo_2_code_evaluators()
    demo_3_llm_as_judge()
    demo_4_custom_judge()
    demo_5_langsmith_evaluation()
    demo_6_rag_evaluation()
    demo_7_prompt_regression()

    print("=" * 60)
    print("That's it! Now open starter.py and build the full eval pipeline.")
    print("=" * 60)
