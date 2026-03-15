# Layer 4: Agents & Orchestration Workshop

A hands-on workshop on building AI agents with LangChain, LangGraph, LangSmith, and LLM Evaluation.

**Duration:** 6 hours (3 days x 2 hours)
**Format:** Instructor-led with follow-along coding exercises

---

## What You'll Build

| Exercise | Branch | What You Build |
|----------|--------|----------------|
| 1. LangChain Agent | `01-langchain-agent` | A Research Assistant with web search, calculator, and Wikipedia tools |
| 2. LangGraph Agent | `02-langgraph-agent` | A stateful Blog Writer with planning, research, writing, and review stages |
| 3. LangSmith | `03-langsmith` | Tracing, evaluation datasets, LLM-as-judge evaluators, and prompt management |
| 4. LLM Evaluation | `04-llm-evaluation` | Golden datasets, LLM-as-judge, adversarial testing, RAG evaluation, and prompt regression |

---

## Schedule

### Day 1 (2 hours)

| Time | Activity |
|------|----------|
| 0:00 - 0:15 | Welcome, setup verification |
| 0:15 - 0:30 | Conceptual: What are agents? ReAct pattern, tool-calling |
| 0:30 - 1:15 | **Hands-on 1:** LangChain Agent |
| 1:15 - 1:20 | Break |
| 1:20 - 1:35 | Conceptual: LangGraph — why stateful workflows? |
| 1:35 - 2:00 | **Hands-on 2 (Part 1):** LangGraph Agent |

### Day 2 (2 hours)

| Time | Activity |
|------|----------|
| 0:00 - 0:10 | Day 1 recap, Q&A |
| 0:10 - 0:50 | **Hands-on 2 (Part 2):** Complete LangGraph Agent |
| 0:50 - 0:55 | Break |
| 0:55 - 1:05 | Conceptual: LangSmith — why observability matters |
| 1:05 - 1:50 | **Hands-on 3:** LangSmith |
| 1:50 - 2:00 | Wrap-up, further resources, Q&A |

### Day 3 (2 hours)

| Time | Activity |
|------|----------|
| 0:00 - 0:10 | Day 2 recap, Q&A |
| 0:10 - 0:30 | Conceptual: LLM Evaluation — why traditional testing breaks |
| 0:30 - 1:15 | **Hands-on 4 (Part 1):** Golden datasets, LLM-as-judge, adversarial testing |
| 1:15 - 1:20 | Break |
| 1:20 - 1:50 | **Hands-on 4 (Part 2):** RAG evaluation with RAGAS, prompt regression testing |
| 1:50 - 2:00 | Wrap-up, further resources, Q&A |

---

## Pre-Workshop Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd Layer4
```

### 2. Create a Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your API keys

You need three free API keys:

| Service | URL | Notes |
|---------|-----|-------|
| OpenAI | https://platform.openai.com/api-keys | Powers the LLM calls |
| Tavily | https://app.tavily.com | Free tier: 1000 searches/month |
| LangSmith | https://smith.langchain.com | Free tier: 5000 traces/month |

### 5. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 6. Verify your setup

```bash
python -c "from langchain_openai import ChatOpenAI; print('LangChain OK')"
python -c "from langgraph.graph import StateGraph; print('LangGraph OK')"
python -c "import langsmith; print('LangSmith OK')"
python -c "from openevals.llm import create_llm_as_judge; print('OpenEvals OK')"
python -c "from ragas.metrics import DiscreteMetric; print('RAGAS OK')"
```

---

## How to Follow Along

Each exercise lives on its own branch. When the instructor says to start an exercise:

```bash
git checkout 01-langchain-agent   # Exercise 1
git checkout 02-langgraph-agent   # Exercise 2
git checkout 03-langsmith         # Exercise 3
git checkout 04-llm-evaluation    # Exercise 4
```

Each branch has:
- `exercises/<topic>/README.md` — step-by-step instructions
- `exercises/<topic>/starter.py` — skeleton code with `# TODO:` markers for you to fill in
- `exercises/<topic>/solution.py` — completed code (peek if you get stuck)

---

## Prerequisites

- Python 3.10+
- Basic Python proficiency
- Familiarity with REST APIs and JSON
- A code editor (VS Code recommended)
- Layers 1-3 of the AI Upskilling track (helpful but not strictly required)
