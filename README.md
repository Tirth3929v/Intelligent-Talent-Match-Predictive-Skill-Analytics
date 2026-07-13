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
  - ❌ **Irrelevant** (Not needed for the role)
  - 📈 **Action Plan** (Crucial skills to learn)
- **Visual Dashboards:** Generates beautiful bar and pie charts to quantify your skill gap instantly.

### 🧭 AI Role Discovery
- **Reverse-Engineer Your Career:** Input your current skills and let the AI predict the top recommended job roles you are most qualified for right now.

### 📥 Enterprise Export
- **Downloadable Insights:** Users can export their personalized Action Plans and Role Recommendations as CSV files for offline tracking and resume building.

---

## 🧠 How the AI Works (Under the Hood)

The core engine of SkillSync AI relies on `scikit-learn` to process and compare textual skill data:

1. **🧹 Data Normalization:** Raw skills from the JSON dataset are aggregated, cleaned, and normalized to remove redundant qualifiers (e.g., "basic", "expert").
2. **🔢 TF-IDF Vectorization:** We use `TfidfVectorizer` to convert skill text into numerical vectors, assigning importance to skills based on their frequency across different job profiles.
3. **📐 Cosine Similarity:** The algorithm computes the cosine similarity between the user's skill vectors and the target role's skill vectors. A semantic match threshold (e.g., `> 0.6`) accurately detects overlapping competencies even if the exact wording differs slightly.

---

## 🛠️ Tech Stack

| Domain | Technology |
|---|---|
| **Frontend & UI** | Streamlit (Python) |
| **Machine Learning / NLP** | Scikit-Learn (`TfidfVectorizer`, `cosine_similarity`) |
| **Data Processing** | Pandas, NumPy, JSON |
| **Data Visualization** | Matplotlib |

---

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

## 📂 Project Structure

```text
📁 Intelligent-Talent-Match
├── 📄 app.py               # Main Streamlit web application and UI logic
├── 📄 job_dataset.json     # Dataset containing industry roles, levels, and required skills
├── 📄 requirements.txt     # Python dependencies
└── 📄 README.md            # Project documentation (You are here!)
```

<div align="center">
  <i>If you find this project helpful, please give it a ⭐ on GitHub!</i>
</div>
