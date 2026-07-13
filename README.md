🚀 SkillSync AI

SkillSync AI is an intelligent career analytics platform that leverages Natural Language Processing (NLP) to bridge the gap between job seekers and industry requirements.

Built for the Hacklabify Hackathon (AI/ML Track).

📌 The Problem

Job seekers and students often struggle to understand exactly which skills they lack for specific career paths. Traditional job boards and career sites don't provide personalized, data-driven insights or actionable learning paths based on a user's existing knowledge base.

💡 The Solution

SkillSync AI dynamically analyzes a user's current skill set against a comprehensive industry dataset to generate a Predictive Skill Gap Analysis. By utilizing advanced text vectorization, it mathematically calculates placement probability and recommends the most optimal job roles for a user's unique profile.

✨ Core Features

🎯 Target Role Analyzer (Gap Analysis)

Calculates a precise "Match Score" for any target job role.

Categorizes skills into: Verified (Matched), Irrelevant, and Action Plan (To Learn).

Generates visual dashboards (bar and pie charts) to quantify the skill gap.

🧭 AI Role Discovery

Reverse-engineers the job search process by taking a user's current skills and predicting the top recommended job roles they are most qualified for.

📥 Enterprise Export

Users can download their personalized Action Plans and Role Recommendations as CSV files for offline tracking.

🧠 How the AI Works (Under the Hood)

The core engine of SkillSync AI relies on Scikit-Learn to process and compare textual skill data:

Data Normalization: Raw skills from the JSON dataset are aggregated, cleaned, and normalized to remove redundant qualifiers (e.g., "basic", "expert").

TF-IDF Vectorization: We use TfidfVectorizer to convert skill text into numerical vectors, assigning importance to skills based on their frequency across different job profiles.

Cosine Similarity: The algorithm computes the cosine similarity between the user's skill vectors and the target role's skill vectors. A semantic match threshold (e.g., > 0.6) is used to accurately detect overlapping competencies even if the exact wording differs slightly.

🛠️ Tech Stack

Frontend & UI: Streamlit (Python)

Machine Learning / NLP: Scikit-Learn (TfidfVectorizer, cosine_similarity)

Data Processing: Pandas, NumPy, JSON

Data Visualization: Matplotlib

🚀 How to Run Locally

Clone the repository:

git clone [https://github.com/yourusername/skillsync-ai.git](https://github.com/yourusername/skillsync-ai.git)
cd skillsync-ai


Install the required dependencies:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run app.py


📂 Project Structure

app.py: The main Streamlit web application and UI logic.

job_dataset.json: The dataset containing industry roles, experience levels, and required skills.

requirements.txt: Python dependencies required to run the project.
