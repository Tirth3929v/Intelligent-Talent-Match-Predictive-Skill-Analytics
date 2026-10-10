<div align="center">
  
# 🚀 SkillSync AI

**Intelligent Talent Match & Predictive Skill Analytics**

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*SkillSync AI is an intelligent career analytics platform that leverages Natural Language Processing (NLP) to bridge the gap between job seekers and industry requirements.* <br>
Built for the **Hacklabify Hackathon (AI/ML Track)**.

</div>

---

## 📌 The Problem

Job seekers and students often struggle to understand exactly **which skills they lack** for specific career paths. Traditional job boards and career sites are static—they don't provide personalized, data-driven insights or actionable learning paths based on a user's existing knowledge base.

## 💡 The Solution

**SkillSync AI** dynamically analyzes a user's current skill set against a comprehensive industry dataset to generate a **Predictive Skill Gap Analysis**. 
By utilizing advanced text vectorization, it mathematically calculates placement probability and recommends the most optimal job roles tailored to a user's unique profile.

---

## ✨ Core Features

### 🎯 Target Role Analyzer (Gap Analysis)
- **Match Score:** Calculates a precise percentage match for any target job role.
- **Skill Categorization:** Automatically sorts your skills into: 
  - ✅ **Verified** (Matched)
  - ↔️ **Transferable / Secondary**
  - 📈 **Action Plan** (Crucial skills to learn)
- **Live Market Analysis:** Grounds target-role recommendations in current hiring research.
- **Visual Dashboards:** Generates bar and pie charts to summarize your skill gap.

### 🧭 AI Role Discovery
- **Reverse-Engineer Your Career:** Input your current skills and let the AI predict the top recommended job roles you are most qualified for right now.

### 📥 Enterprise Export
- **Downloadable Insights:** Users can export their personalized Action Plans and Role Recommendations as CSV files for offline tracking and resume building.

---

## 🧠 How the Analysis Works

The Target Role Analyzer uses the agentic FastAPI pipeline:

1. **Live market grounding:** Tavily searches for current role requirements and returns source URLs.
2. **Structured analysis:** An OpenAI-compatible model (Nebius by default; Gemini is configurable) evaluates the candidate profile against the search results.
3. **Validated response:** FastAPI validates the match score, skill categories, learning roadmap, interview questions, and sources before returning them to Streamlit.

The separate AI Role Discovery tab continues to use the local dataset and scikit-learn TF-IDF matching.

---

## 🛠️ Tech Stack

| Domain | Technology |
|---|---|
| **Frontend & UI** | Streamlit (Python) |
| **Agentic analysis** | FastAPI, Tavily Search, OpenAI-compatible model API |
| **Role discovery** | Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`) |
| **Data Processing** | Pandas, NumPy, JSON |
| **Data Visualization** | Matplotlib |

---

## Agentic FastAPI API

The FastAPI backend grounds each analysis in current Tavily search results and uses
an OpenAI-compatible model endpoint for structured skill-gap analysis. By default,
it uses Nebius Token Factory; configure `OPENAI_BASE_URL`, `NEBIUS_API_KEY`, and
`NEBIUS_MODEL` to use another compatible provider, such as Google Gemini.

```text
Streamlit UI ──> FastAPI Backend ──┬──> Tavily AI (Real-time Market Demands)
                                   └──> Nebius / LLM (Synthesis & Actionable Roadmap)
```

1. Install dependencies with `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and set `TAVILY_API_KEY` and `NEBIUS_API_KEY`.
   `OPENAI_BASE_URL` and `NEBIUS_MODEL` can optionally select a different compatible
   endpoint and model. The `NEBIUS_API_KEY` variable is used as the API credential
   regardless of provider.
3. Start the API from the project directory:

   ```bash
   uvicorn skillsync_api.main:app --host 0.0.0.0 --port 8000
   ```

The API exposes `GET /health` and `POST /api/analyze-gap`. Interactive API
documentation is available at `/docs`. The analysis endpoint expects JSON with
`current_role`, `current_level`, `user_skills` (an array of strings), and
`target_role`.

The Streamlit **Target Role Analyzer** calls this API. Start the API first, then
run `streamlit run app.py` in a second terminal. Set `SKILLSYNC_API_BASE_URL` if
the API is hosted somewhere other than `http://127.0.0.1:8000`. The separate
**AI Role Discovery** tab continues to use the local dataset.

## 🚀 How to Run Locally

Get up and running in a few simple steps!

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tirth3929v/Intelligent-Talent-Match-Predictive-Skill-Analytics.git
   cd Intelligent-Talent-Match-Predictive-Skill-Analytics
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 🔒 Enterprise Zero-Trust Security Architecture

SkillSync AI implements defense-in-depth across the API and agentic pipelines:

1. **Header-Based Authentication:** Backend analysis endpoints are secured with `X-API-Key` authentication (`verify_internal_auth`), preventing unauthorized usage.
2. **Prompt Injection & Jailbreak Defense:** String payloads are bounded, cleansed of control/meta-characters (`<>{}[\]\\`), and scanned against heuristic jailbreak signatures (`ignore previous`, `system prompt`, `you are now`, `drop table`).
3. **Strict CORS & Host Protection:** Configurable origin binding limits browser traffic exclusively to trusted clients (Streamlit frontend).
4. **Denial-of-Wallet & Rate Limiting:** Built-in `slowapi` rate limiting (10 requests/minute/IP) prevents malicious upstream API quota exhaustion.
5. **Defensive System Prompt Wrappers:** System prompts enforce strict boundary markers (`UNTRUSTED_DATA_BOUNDARY`) and explicitly instruct the LLM to ignore meta-instructions in user input or search results.

---

## 🎯 Grounded >90% Analysis Accuracy & Deterministic Engine

1. **Two-Stage Canonical Taxonomy Mapping:** User skills are normalized and mapped through a canonical taxonomy of 500+ standard tech skills with automatic synonym resolution (`js` $\rightarrow$ `javascript`, `postgres` $\rightarrow$ `postgresql`, `k8s` $\rightarrow$ `kubernetes`).
2. **Grounding Verification (Hallucination Guardrail):** Missing and matched critical skills are verified against the live Tavily search corpus and canonical taxonomy. Hallucinated tools or frameworks are dropped before returning JSON.
3. **Deterministic Scoring Engine:** Eliminates LLM arithmetic drift by setting `temperature = 0.0` and computing match percentages with a deterministic scoring formula:
   $$\text{Score} = \min\left(100, \max\left(0, \operatorname{round}\left(\frac{\vert\text{Matched}\vert + 0.35 \times \vert\text{Transferable}\vert}{\vert\text{Matched}\vert + \vert\text{Missing}\vert + 0.35 \times \vert\text{Transferable}\vert} \times 100\right)\right)\right)$$
4. **Evaluation Benchmark Suite (`benchmark_eval.json`):** 15 reference career profiles benchmarked with automated precision and recall evaluation:
   - **Micro-Average Precision:** 100.00%
   - **Micro-Average Recall:** 97.46%
   - **Macro-Average Precision:** 100.00%
   - **Macro-Average Recall:** 97.26%
   - **Arithmetic Reproducibility:** 100.00%

To execute the benchmark suite:
```bash
python benchmark_runner.py
```

---

## 📂 Project Structure

```text
📁 Intelligent-Talent-Match
├── 📄 app.py                        # Streamlit web application & UI logic
├── 📄 benchmark_eval.json           # 15 reference career benchmark profiles
├── 📄 benchmark_runner.py           # Benchmark accuracy & precision test runner
├── 📄 job_dataset.json              # Industry roles and skill database
├── 📄 requirements.txt              # Project dependencies
├── 📁 skillsync_api/
│   ├── 📄 main.py                   # FastAPI application with CORS & rate limiting
│   ├── 📄 security.py               # Zero-trust auth, sanitization & injection defense
│   ├── 📄 evaluator.py              # Canonical taxonomy, synonym mapper & deterministic scorer
│   ├── 📄 models.py                 # Pydantic schemas
│   └── 📄 service.py                # Tavily search & grounded analysis coordinator
└── 📁 tests/
    ├── 📄 test_skillsync_api.py     # API service unit tests
    ├── 📄 test_security_and_evaluator.py # Security & accuracy benchmark tests
    └── 📄 test_streamlit_ui.py      # Streamlit UI smoke tests
```

<div align="center">
  <i>If you find this project helpful, please give it a ⭐ on GitHub!</i>
</div>

