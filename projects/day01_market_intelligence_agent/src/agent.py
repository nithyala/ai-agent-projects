"""
╔══════════════════════════════════════════════════════════════╗
║   Day 01 — Market Intelligence Multi-Agent System            ║
║   Built by: Nithya Jambulingam                               ║
║   Stack: LangGraph · LangChain · OpenAI GPT-4o · Python      ║
╚══════════════════════════════════════════════════════════════╝

4 specialized AI agents work in sequence:
  Agent 1 — ResearchAgent   : Fetches deep research on any topic
  Agent 2 — SentimentAgent  : Scores tone, confidence, key drivers
  Agent 3 — InsightAgent    : Extracts 5 actionable business insights
  Agent 4 — ReportAgent     : Compiles full markdown + HTML report

Run:
  python src/agent.py "OpenAI enterprise market 2025"
  python src/agent.py "Anthropic vs OpenAI 2025"
  python src/agent.py "Electric vehicle battery supply chain"
"""

import os
import sys
import json
import time
import textwrap
from datetime import datetime
from typing import TypedDict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END


# ══════════════════════════════════════════════════════════════
# TERMINAL COLORS
# ══════════════════════════════════════════════════════════════
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    MAGENTA= "\033[95m"
    RED    = "\033[91m"
    WHITE  = "\033[97m"
    DIM    = "\033[2m"


def banner():
    print(f"""
{C.CYAN}{C.BOLD}
╔══════════════════════════════════════════════════════════════╗
║       🤖  MARKET INTELLIGENCE MULTI-AGENT SYSTEM            ║
║       Built by Nithya Jambulingam  |  LangGraph + GPT-4o    ║
╚══════════════════════════════════════════════════════════════╝
{C.RESET}""")


def print_agent_header(num: int, name: str, role: str, color: str):
    print(f"\n{color}{C.BOLD}{'─'*60}")
    print(f"  Agent {num}/4 — {name}")
    print(f"  Role: {role}")
    print(f"{'─'*60}{C.RESET}")


def print_streaming(label: str, text: str, color: str):
    """Simulate streaming output word by word."""
    print(f"\n{color}{C.BOLD}{label}:{C.RESET}")
    words = text.split()
    line = ""
    for i, word in enumerate(words):
        line += word + " "
        print(f"\r  {line}", end="", flush=True)
        if (i + 1) % 12 == 0:
            print()
            line = ""
        time.sleep(0.02)
    if line.strip():
        print(f"\r  {line}")


def spinner(msg: str, duration: float = 1.2):
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    end_time = time.time() + duration
    i = 0
    while time.time() < end_time:
        print(f"\r  {C.YELLOW}{frames[i % len(frames)]}{C.RESET}  {msg}", end="", flush=True)
        time.sleep(0.08)
        i += 1
    print(f"\r  {C.GREEN}✓{C.RESET}  {msg}{' '*10}")


# ══════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════
class AgentState(TypedDict):
    topic: str
    raw_research: str
    sentiment_json: str
    sentiment_label: str
    sentiment_score: int
    sentiment_drivers: List[str]
    insights: List[str]
    final_report_md: str
    final_report_html: str
    timestamp: str
    output_path: str


# ══════════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════════
def get_llm(temperature: float = 0.3) -> ChatOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"\n{C.RED}❌ ERROR: OPENAI_API_KEY not set.{C.RESET}")
        print(f"  Run: {C.YELLOW}export OPENAI_API_KEY=your_key_here{C.RESET}\n")
        sys.exit(1)
    return ChatOpenAI(model="gpt-4o", temperature=temperature, api_key=api_key)


# ══════════════════════════════════════════════════════════════
# AGENT 1 — RESEARCH
# ══════════════════════════════════════════════════════════════
def research_agent(state: AgentState) -> AgentState:
    print_agent_header(1, "ResearchAgent", "Senior Market Research Analyst", C.CYAN)
    spinner(f"Researching: '{state['topic']}'", 1.5)

    llm = get_llm(temperature=0.4)
    messages = [
        SystemMessage(content="""You are a senior market research analyst at a top-tier consulting firm.
Given a company, product, or market topic, provide a comprehensive research summary including:
- Executive Overview (2-3 sentences)
- Background & History
- Recent Developments (last 6-12 months)
- Key Players & Competitors
- Market Position & Size
- Challenges & Opportunities

Be factual, specific, and use data/numbers where possible. Write 350-400 words."""),
        HumanMessage(content=f"Research topic: {state['topic']}")
    ]

    response = llm.invoke(messages)
    state["raw_research"] = response.content
    print_streaming("📋 Research Summary", state["raw_research"], C.CYAN)
    print(f"\n  {C.GREEN}✅ ResearchAgent complete{C.RESET}")
    return state


# ══════════════════════════════════════════════════════════════
# AGENT 2 — SENTIMENT
# ══════════════════════════════════════════════════════════════
def sentiment_agent(state: AgentState) -> AgentState:
    print_agent_header(2, "SentimentAgent", "NLP & Sentiment Analysis Expert", C.MAGENTA)
    spinner("Analyzing sentiment and tone...", 1.2)

    llm = get_llm(temperature=0.1)
    messages = [
        SystemMessage(content="""You are an expert NLP and sentiment analysis specialist.
Analyze the following research text and return ONLY a valid JSON object with exactly these keys:
{
  "sentiment": "Positive" or "Neutral" or "Negative",
  "confidence": <integer 0-100>,
  "drivers": ["driver 1", "driver 2", "driver 3"],
  "tone": "one word describing the overall tone",
  "outlook": "Short-term outlook: one sentence"
}
Return ONLY the JSON. No explanation, no markdown."""),
        HumanMessage(content=state["raw_research"])
    ]

    response = llm.invoke(messages)
    raw = response.content.strip().strip("```json").strip("```").strip()
    data = json.loads(raw)

    state["sentiment_json"] = json.dumps(data, indent=2)
    state["sentiment_label"] = data.get("sentiment", "Neutral")
    state["sentiment_score"] = data.get("confidence", 50)
    state["sentiment_drivers"] = data.get("drivers", [])

    emoji = "🟢" if state["sentiment_label"] == "Positive" else "🟡" if state["sentiment_label"] == "Neutral" else "🔴"
    print(f"\n  {C.MAGENTA}{C.BOLD}📊 Sentiment Result:{C.RESET}")
    print(f"  {emoji} Overall:    {C.BOLD}{state['sentiment_label']}{C.RESET}")
    print(f"  📈 Confidence:  {C.BOLD}{state['sentiment_score']}%{C.RESET}")
    print(f"  🎭 Tone:        {C.BOLD}{data.get('tone', 'N/A')}{C.RESET}")
    print(f"  🔮 Outlook:     {data.get('outlook', 'N/A')}")
    print(f"\n  {C.MAGENTA}Key Sentiment Drivers:{C.RESET}")
    for d in state["sentiment_drivers"]:
        print(f"    • {d}")

    print(f"\n  {C.GREEN}✅ SentimentAgent complete{C.RESET}")
    return state


# ══════════════════════════════════════════════════════════════
# AGENT 3 — INSIGHT
# ══════════════════════════════════════════════════════════════
def insight_agent(state: AgentState) -> AgentState:
    print_agent_header(3, "InsightAgent", "Strategic Business Consultant", C.YELLOW)
    spinner("Extracting actionable insights...", 1.3)

    llm = get_llm(temperature=0.5)
    messages = [
        SystemMessage(content="""You are a senior strategic business consultant at McKinsey.
From the research provided, extract exactly 5 high-value, actionable business insights.
Each insight must be:
- Specific and data-driven (not generic)
- Actionable (what should a business DO with this info?)
- 1-2 sentences max

Return ONLY a valid JSON array of 5 strings. No other text."""),
        HumanMessage(content=state["raw_research"])
    ]

    response = llm.invoke(messages)
    raw = response.content.strip().strip("```json").strip("```").strip()
    insights = json.loads(raw)
    state["insights"] = insights

    print(f"\n  {C.YELLOW}{C.BOLD}💡 Strategic Insights:{C.RESET}")
    for i, insight in enumerate(insights, 1):
        wrapped = textwrap.fill(insight, width=70, subsequent_indent="       ")
        print(f"\n  {C.BOLD}  {i}.{C.RESET} {wrapped}")

    print(f"\n  {C.GREEN}✅ InsightAgent complete{C.RESET}")
    return state


# ══════════════════════════════════════════════════════════════
# AGENT 4 — REPORT
# ══════════════════════════════════════════════════════════════
def report_agent(state: AgentState) -> AgentState:
    print_agent_header(4, "ReportAgent", "Executive Report Compiler", C.BLUE)
    spinner("Compiling final report...", 1.0)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    state["timestamp"] = ts

    sentiment_emoji = "🟢" if state["sentiment_label"] == "Positive" else "🟡" if state["sentiment_label"] == "Neutral" else "🔴"
    insights_md = "\n".join([f"{i+1}. {ins}" for i, ins in enumerate(state["insights"])])
    drivers_md = "\n".join([f"- {d}" for d in state["sentiment_drivers"]])

    # ── Markdown Report ──
    state["final_report_md"] = f"""# 📊 Market Intelligence Report

**Topic:** {state['topic']}
**Generated:** {ts}
**Pipeline:** ResearchAgent → SentimentAgent → InsightAgent → ReportAgent

---

## 🔍 Research Summary

{state['raw_research']}

---

## 💬 Sentiment Analysis

| Field | Value |
|-------|-------|
| Overall Sentiment | {sentiment_emoji} {state['sentiment_label']} |
| Confidence Score | {state['sentiment_score']}% |

**Key Sentiment Drivers:**
{drivers_md}

---

## 💡 Strategic Insights

{insights_md}

---

## 🤖 About This Pipeline

This report was autonomously generated by a **4-agent LangGraph pipeline**.

```
ResearchAgent → SentimentAgent → InsightAgent → ReportAgent
     │                │               │               │
  GPT-4o           GPT-4o          GPT-4o         Python
  analyst          NLP expert      consultant     compiler
```

*Generated by Nithya Jambulingam's AI Agent Projects — Day 01*
"""

    # ── HTML Report ──
    insights_html = "".join([
        f'<div class="insight"><span class="insight-num">{i+1}</span><p>{ins}</p></div>'
        for i, ins in enumerate(state["insights"])
    ])
    drivers_html = "".join([f'<li>{d}</li>' for d in state["sentiment_drivers"]])
    bar_color = "#22c55e" if state["sentiment_label"] == "Positive" else "#eab308" if state["sentiment_label"] == "Neutral" else "#ef4444"
    sentiment_color = "#166534" if state["sentiment_label"] == "Positive" else "#854d0e" if state["sentiment_label"] == "Neutral" else "#991b1b"
    sentiment_bg = "#dcfce7" if state["sentiment_label"] == "Positive" else "#fef9c3" if state["sentiment_label"] == "Neutral" else "#fee2e2"

    state["final_report_html"] = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Market Intelligence Report — {state['topic']}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; }}

  .hero {{ background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 50%, #1a1a2e 100%); padding: 60px 40px 40px; text-align: center; border-bottom: 1px solid #334155; }}
  .hero .badge {{ display: inline-block; background: #3b82f6; color: white; font-size: 11px; font-weight: 700; padding: 4px 14px; border-radius: 20px; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 20px; }}
  .hero h1 {{ font-size: 2.2rem; font-weight: 800; color: white; margin-bottom: 10px; line-height: 1.3; }}
  .hero .meta {{ color: #94a3b8; font-size: 14px; margin-top: 8px; }}
  .hero .pipeline {{ display: flex; justify-content: center; align-items: center; gap: 8px; margin-top: 28px; flex-wrap: wrap; }}
  .agent-badge {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 16px; font-size: 13px; font-weight: 600; color: #93c5fd; }}
  .arrow {{ color: #475569; font-size: 18px; }}

  .container {{ max-width: 900px; margin: 0 auto; padding: 40px 24px; }}

  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 16px; padding: 32px; margin-bottom: 28px; }}
  .card-title {{ font-size: 18px; font-weight: 700; color: #93c5fd; margin-bottom: 20px; display: flex; align-items: center; gap: 10px; }}
  .card-body {{ color: #cbd5e1; line-height: 1.8; font-size: 15px; }}

  .sentiment-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .sentiment-box {{ background: #0f172a; border-radius: 12px; padding: 20px; text-align: center; }}
  .sentiment-label {{ font-size: 28px; font-weight: 800; }}
  .sentiment-sub {{ font-size: 12px; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }}
  .score-bar {{ background: #0f172a; border-radius: 12px; padding: 20px; }}
  .bar-track {{ background: #334155; border-radius: 99px; height: 12px; margin: 12px 0; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 99px; transition: width 1s ease; background: {bar_color}; width: {state['sentiment_score']}%; }}
  .badge-sentiment {{ display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 13px; font-weight: 700; background: {sentiment_bg}; color: {sentiment_color}; }}

  .drivers {{ margin-top: 16px; }}
  .drivers li {{ padding: 10px 0; border-bottom: 1px solid #334155; color: #94a3b8; font-size: 14px; list-style: none; padding-left: 20px; position: relative; }}
  .drivers li::before {{ content: "→"; position: absolute; left: 0; color: #3b82f6; }}

  .insight {{ display: flex; gap: 16px; align-items: flex-start; padding: 16px 0; border-bottom: 1px solid #334155; }}
  .insight:last-child {{ border-bottom: none; }}
  .insight-num {{ background: #3b82f6; color: white; font-weight: 800; font-size: 13px; min-width: 28px; height: 28px; border-radius: 8px; display: flex; align-items: center; justify-content: center; }}
  .insight p {{ color: #cbd5e1; line-height: 1.7; font-size: 14px; }}

  .pipeline-visual {{ display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; }}
  .agent-node {{ flex: 1; min-width: 180px; background: #0f172a; border: 1px solid #334155; border-radius: 12px; padding: 18px; text-align: center; }}
  .agent-node .icon {{ font-size: 28px; margin-bottom: 8px; }}
  .agent-node .name {{ font-weight: 700; font-size: 13px; color: #93c5fd; }}
  .agent-node .role {{ font-size: 11px; color: #64748b; margin-top: 4px; }}

  footer {{ text-align: center; padding: 32px; color: #475569; font-size: 13px; border-top: 1px solid #1e293b; }}
  footer a {{ color: #3b82f6; text-decoration: none; }}
</style>
</head>
<body>

<div class="hero">
  <div class="badge">🤖 AI-Generated Report</div>
  <h1>{state['topic']}</h1>
  <div class="meta">Generated: {ts} &nbsp;|&nbsp; 4-Agent LangGraph Pipeline &nbsp;|&nbsp; OpenAI GPT-4o</div>
  <div class="pipeline">
    <div class="agent-badge">🔍 ResearchAgent</div>
    <div class="arrow">→</div>
    <div class="agent-badge">💬 SentimentAgent</div>
    <div class="arrow">→</div>
    <div class="agent-badge">💡 InsightAgent</div>
    <div class="arrow">→</div>
    <div class="agent-badge">📄 ReportAgent</div>
  </div>
</div>

<div class="container">

  <div class="card">
    <div class="card-title">🔍 Research Summary</div>
    <div class="card-body">{state['raw_research'].replace(chr(10), '<br>')}</div>
  </div>

  <div class="card">
    <div class="card-title">💬 Sentiment Analysis</div>
    <div class="sentiment-grid">
      <div class="sentiment-box">
        <div class="sentiment-label">{sentiment_emoji} {state['sentiment_label']}</div>
        <div class="sentiment-sub">Overall Sentiment</div>
        <br><span class="badge-sentiment">{state['sentiment_label']}</span>
      </div>
      <div class="score-bar">
        <div class="sentiment-sub">Confidence Score</div>
        <div style="font-size:32px;font-weight:800;color:white;margin-top:8px">{state['sentiment_score']}%</div>
        <div class="bar-track"><div class="bar-fill"></div></div>
      </div>
    </div>
    <div class="card-title" style="font-size:14px;margin-bottom:8px">Key Sentiment Drivers</div>
    <ul class="drivers">{drivers_html}</ul>
  </div>

  <div class="card">
    <div class="card-title">💡 Strategic Insights</div>
    {insights_html}
  </div>

  <div class="card">
    <div class="card-title">🤖 Pipeline Architecture</div>
    <div class="pipeline-visual">
      <div class="agent-node"><div class="icon">🔍</div><div class="name">ResearchAgent</div><div class="role">GPT-4o Analyst</div></div>
      <div class="agent-node"><div class="icon">💬</div><div class="name">SentimentAgent</div><div class="role">NLP Expert</div></div>
      <div class="agent-node"><div class="icon">💡</div><div class="name">InsightAgent</div><div class="role">Strategy Consultant</div></div>
      <div class="agent-node"><div class="icon">📄</div><div class="name">ReportAgent</div><div class="role">Report Compiler</div></div>
    </div>
    <div style="margin-top:20px;color:#64748b;font-size:13px;line-height:1.8">
      Each agent is a specialized <strong style="color:#93c5fd">LangGraph node</strong> with its own system prompt and role.
      State flows through the graph: <code style="background:#0f172a;padding:2px 8px;border-radius:4px;color:#fbbf24">AgentState</code> is enriched at each step.
      No human intervention required — fully autonomous pipeline.
    </div>
  </div>

</div>

<footer>
  Built by <a href="https://github.com/nithyala/ai-agent-projects">Nithya Jambulingam</a> &nbsp;·&nbsp;
  AI Agent Projects — Day 01 &nbsp;·&nbsp;
  LangGraph + OpenAI GPT-4o
</footer>

</body>
</html>"""

    # Save files
    os.makedirs("output", exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M")
    md_path = f"output/report_{ts_file}.md"
    html_path = f"output/report_{ts_file}.html"

    with open(md_path, "w") as f:
        f.write(state["final_report_md"])
    with open(html_path, "w") as f:
        f.write(state["final_report_html"])

    state["output_path"] = html_path
    print(f"\n  {C.BLUE}{C.BOLD}📄 Report compiled successfully!{C.RESET}")
    print(f"  {C.GREEN}✅ ReportAgent complete{C.RESET}")
    return state


# ══════════════════════════════════════════════════════════════
# PIPELINE BUILDER
# ══════════════════════════════════════════════════════════════
def build_pipeline():
    graph = StateGraph(AgentState)
    graph.add_node("research", research_agent)
    graph.add_node("sentiment", sentiment_agent)
    graph.add_node("insight", insight_agent)
    graph.add_node("report", report_agent)
    graph.set_entry_point("research")
    graph.add_edge("research", "sentiment")
    graph.add_edge("sentiment", "insight")
    graph.add_edge("insight", "report")
    graph.add_edge("report", END)
    return graph.compile()


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run(topic: str):
    banner()
    print(f"  {C.WHITE}{C.BOLD}Topic:{C.RESET} {topic}")
    print(f"  {C.DIM}Starting 4-agent pipeline...{C.RESET}")

    start = time.time()
    pipeline = build_pipeline()

    initial_state: AgentState = {
        "topic": topic,
        "raw_research": "",
        "sentiment_json": "",
        "sentiment_label": "",
        "sentiment_score": 0,
        "sentiment_drivers": [],
        "insights": [],
        "final_report_md": "",
        "final_report_html": "",
        "timestamp": "",
        "output_path": ""
    }

    result = pipeline.invoke(initial_state)
    elapsed = round(time.time() - start, 1)

    print(f"""
{C.GREEN}{C.BOLD}
╔══════════════════════════════════════════════════════════════╗
║   ✅  PIPELINE COMPLETE                                      ║
╠══════════════════════════════════════════════════════════════╣
║   ⏱  Time elapsed : {elapsed}s{' '*(38-len(str(elapsed)))}║
║   📄  Markdown    : output/report_*.md                       ║
║   🌐  HTML Report : {result['output_path'][:38]+'...' if len(result['output_path'])>38 else result['output_path']:<41}║
╠══════════════════════════════════════════════════════════════╣
║   👉  Open the HTML file in your browser to share!           ║
╚══════════════════════════════════════════════════════════════╝
{C.RESET}""")

    return result


if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "OpenAI and the enterprise AI market in 2025"
    run(topic)
