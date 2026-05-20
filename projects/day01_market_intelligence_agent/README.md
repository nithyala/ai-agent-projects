# 🤖 Day 01 — Market Intelligence Multi-Agent System

> **Built by:** Nithya Jambulingam | **Stack:** LangGraph · LangChain · OpenAI GPT-4o · Python

---

## 🎯 What This Does

A **4-agent autonomous pipeline** that researches any company or market topic and produces:
- A beautiful **HTML report** (open in browser, perfect for screen sharing)
- A **Markdown report** saved to `/output/`
- Live animated terminal output showing each agent working in real time

```
python src/agent.py "Anthropic vs OpenAI 2025"
```

---

## 🏗️ Architecture

```
User Input (topic)
       │
       ▼
┌──────────────────────────────────────────────────┐
│              LangGraph StateGraph                │
│                                                  │
│  ┌─────────────┐    ┌──────────────┐             │
│  │ResearchAgent│───▶│SentimentAgent│             │
│  │  GPT-4o     │    │   GPT-4o     │             │
│  │  Analyst    │    │  NLP Expert  │             │
│  └─────────────┘    └──────┬───────┘             │
│                            │                     │
│  ┌─────────────┐    ┌──────▼───────┐             │
│  │ ReportAgent │◀───│ InsightAgent │             │
│  │   Python    │    │   GPT-4o     │             │
│  │  Compiler   │    │  Consultant  │             │
│  └─────────────┘    └─────────────┘             │
└──────────────────────────────────────────────────┘
       │
       ▼
 📄 output/report_YYYYMMDD_HHMM.html  ← Open in browser!
 📝 output/report_YYYYMMDD_HHMM.md
```

---

## 🧠 How It Works — Step by Step

### 1. Shared State (TypedDict)
All agents communicate through a single typed state object:
```python
class AgentState(TypedDict):
    topic: str              # Input
    raw_research: str       # Added by ResearchAgent
    sentiment_label: str    # Added by SentimentAgent
    sentiment_score: int    # Added by SentimentAgent
    sentiment_drivers: List[str]
    insights: List[str]     # Added by InsightAgent
    final_report_html: str  # Added by ReportAgent
```

### 2. LangGraph Pipeline
Nodes connected with directed edges — no loops, no branching needed:
```python
graph.set_entry_point("research")
graph.add_edge("research", "sentiment")
graph.add_edge("sentiment", "insight")
graph.add_edge("insight", "report")
graph.add_edge("report", END)
```

### 3. Agent 1 — ResearchAgent
- **Role:** Senior Market Research Analyst
- **Does:** Calls GPT-4o with a detailed analyst system prompt
- **Output:** 350-400 word research summary added to state

### 4. Agent 2 — SentimentAgent
- **Role:** NLP & Sentiment Expert
- **Does:** Analyzes the research text, returns structured JSON
- **Output:** `sentiment_label`, `sentiment_score` (0-100), `sentiment_drivers`

### 5. Agent 3 — InsightAgent
- **Role:** McKinsey-style Strategy Consultant
- **Does:** Extracts 5 actionable insights from research
- **Output:** List of 5 specific, data-driven business insights

### 6. Agent 4 — ReportAgent
- **Role:** Pure Python compiler (no LLM needed!)
- **Does:** Assembles all state into a polished HTML + Markdown report
- **Output:** Saves files, prints success summary

---

## 🚀 How to Run

### Setup
```bash
cd projects/day01_market_intelligence_agent
pip install -r requirements.txt
export OPENAI_API_KEY=your_openai_key_here
```

### Run
```bash
# Default topic
python src/agent.py

# Custom topics
python src/agent.py "OpenAI and enterprise AI market 2025"
python src/agent.py "Anthropic Claude vs GPT-4 comparison"
python src/agent.py "Electric vehicle battery supply chain risks"
python src/agent.py "Netflix vs Disney+ streaming wars 2025"
```

### View Output
```bash
# Open the HTML report in your browser
open output/report_*.html          # Mac
start output/report_*.html         # Windows
xdg-open output/report_*.html      # Linux
```

---

## 📊 Sample Terminal Output

```
╔══════════════════════════════════════════════════════════════╗
║       🤖  MARKET INTELLIGENCE MULTI-AGENT SYSTEM            ║
╚══════════════════════════════════════════════════════════════╝

──────────────────────────────────────────────────────────────
  Agent 1/4 — ResearchAgent
  Role: Senior Market Research Analyst
──────────────────────────────────────────────────────────────
  ✓  Researching: 'OpenAI and enterprise AI market 2025'

📋 Research Summary:
  OpenAI has rapidly expanded its enterprise footprint in 2025...

──────────────────────────────────────────────────────────────
  Agent 2/4 — SentimentAgent
──────────────────────────────────────────────────────────────
  📊 Sentiment Result:
  🟢 Overall:    Positive
  📈 Confidence: 84%
  🔮 Outlook:    Strong growth trajectory expected through 2026

✅  PIPELINE COMPLETE — HTML report saved!
```

---

## 🔑 Key Concepts

| Concept | Implementation |
|---------|---------------|
| Multi-agent orchestration | LangGraph `StateGraph` with 4 nodes |
| Typed shared state | `TypedDict` passed and enriched through pipeline |
| Specialized LLM personas | Unique system prompts per agent |
| Structured JSON output | LLM returns JSON, parsed and validated |
| Zero-dependency reporting | ReportAgent uses pure Python — no extra LLM call |
| Rich terminal UX | ANSI colors + streaming animation |

---

## 📁 Structure

```
day01_market_intelligence_agent/
├── src/
│   └── agent.py       # All 4 agents + LangGraph pipeline (~300 lines)
├── output/            # Generated reports (auto-created on run)
│   ├── report_*.html  # 🌐 Open this in browser!
│   └── report_*.md
├── requirements.txt
└── README.md
```

---

*Part of [Nithya's AI Agent Projects](https://github.com/nithyala/ai-agent-projects) — 2 new AI agent projects auto-generated daily.*
