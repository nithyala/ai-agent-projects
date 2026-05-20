# 🤖 Nithya's AI Agent Projects

> **2 new production-quality AI agent projects, auto-generated every day at 6 PM UTC.**
> Built with LangGraph · LangChain · OpenAI GPT-4o · Python · GitHub Actions

---

## 🚀 How This Works

```
GitHub Actions (cron: 6 PM UTC)
         │
         ▼
scripts/generate_daily_projects.py
         │
         ├── Claude API generates 2 unique agent themes
         ├── Full Python source (LangGraph multi-agent pipeline)
         ├── Detailed README with architecture + explanation
         ├── HTML report output (open in browser)
         └── Auto-commits and pushes — zero human intervention
```

---

## 📁 Projects

| Day | Project | Description | Date |
|-----|---------|-------------|------|
| Day 01a | [Market Intelligence Multi-Agent System](projects/day01_market_intelligence_agent) | 4-agent pipeline: research → sentiment → insights → HTML report | 2026-05-19 |
<!-- PROJECTS_TABLE -->

---

## ⚙️ Run Any Project Locally

```bash
git clone https://github.com/nithyala/ai-agent-projects
cd ai-agent-projects/projects/day01_market_intelligence_agent
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python src/agent.py "Your topic here"

# Then open the HTML report in your browser:
open output/report_*.html
```

---

## 🔐 Enable Daily Auto-Generation

Add your Anthropic API key as a GitHub secret:
```
GitHub repo → Settings → Secrets → Actions → New secret
Name:  ANTHROPIC_API_KEY
Value: your-anthropic-api-key
```

The GitHub Actions workflow then runs automatically every day at **6 PM UTC**.

---

## 🧠 Tech Stack

| Tool | Purpose |
|------|---------|
| LangGraph | Multi-agent StateGraph orchestration |
| LangChain | LLM abstraction & prompt management |
| OpenAI GPT-4o | Core LLM for all agents |
| Python | Primary language |
| GitHub Actions | Daily automation (cron) |
| Claude API | Project generation engine |

---

*Built by [Nithya Jambulingam](https://www.linkedin.com/in/nithya-jambulingam) — GenAI Engineer with 5 years experience*
