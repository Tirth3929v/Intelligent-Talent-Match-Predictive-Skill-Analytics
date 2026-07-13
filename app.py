# import json
# import re
# import importlib.util
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from collections import Counter

# if importlib.util.find_spec("streamlit") is not None:
#     st = importlib.import_module("streamlit")
# else:  # pragma: no cover - fallback for environments without streamlit installed
#     class _StreamlitStub:
#         def __init__(self):
#             self.sidebar = self

#         def cache_data(self, func=None, *args, **kwargs):
#             if func is None:
#                 def decorator(f):
#                     return f
#                 return decorator
#             return func

#         def cache_resource(self, func=None, *args, **kwargs):
#             if func is None:
#                 def decorator(f):
#                     return f
#                 return decorator
#             return func

#         def columns(self, count):
#             return tuple(_StreamlitStub() for _ in range(count))

#         def spinner(self, *args, **kwargs):
#             return self

#         def __enter__(self):
#             return self

#         def __exit__(self, exc_type, exc, tb):
#             return False

#         def slider(self, *args, **kwargs):
#             if args:
#                 return args[-1]
#             return kwargs.get("value", kwargs.get("default", 0))

#         def button(self, *args, **kwargs):
#             return False

#         def text_input(self, *args, **kwargs):
#             return ""

#         def selectbox(self, *args, **kwargs):
#             return ""

#         def text_area(self, *args, **kwargs):
#             return ""

#         def metric(self, *args, **kwargs):
#             return None

#         def pyplot(self, *args, **kwargs):
#             return None

#         def dataframe(self, *args, **kwargs):
#             return None

#         def set_page_config(self, *args, **kwargs):
#             return None

#         def __getattr__(self, name):
#             return lambda *args, **kwargs: None

#     st = _StreamlitStub()

# st.set_page_config(page_title="Intelligent Talent Match", page_icon="🎯", layout="wide")

# # ── Load & cache data ──────────────────────────────────────────────────────────
# @st.cache_data
# def load_data():
#     with open("job_dataset.json", "r", encoding="utf-8") as f:
#         job_data = json.load(f)
#     df = pd.DataFrame(job_data)
#     return df

# @st.cache_resource
# def build_models(df):
#     QUALIFIERS = r'\b(basics|fundamentals|advanced|expert|intro|introductory|basic|intermediate)\b'

#     def normalize_skill(skill: str) -> str:
#         skill = skill.lower().strip()
#         skill = re.sub(QUALIFIERS, '', skill)
#         skill = re.sub(r'\s+', ' ', skill).strip()
#         return skill

#     unique_roles = sorted(df['Title'].dropna().unique())

#     def get_role_skill_profile(role_title: str, top_n: int = 20) -> dict:
#         entries = df[df['Title'] == role_title]
#         skill_counter = Counter()
#         for _, row in entries.iterrows():
#             for skill in row.get('Skills', []):
#                 normalized = normalize_skill(skill)
#                 if normalized:
#                     skill_counter[normalized] += 1
#         total = sum(skill_counter.values())
#         top_skills = skill_counter.most_common(top_n)
#         return {
#             'skills': [s for s, _ in top_skills],
#             'weights': {s: c / total for s, c in top_skills}
#         }

#     role_profiles = {role: get_role_skill_profile(role) for role in unique_roles}

#     all_skills = sorted({s for p in role_profiles.values() for s in p['skills']})
#     vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
#     skill_vectors = vectorizer.fit_transform(all_skills)

#     return unique_roles, role_profiles, all_skills, vectorizer, skill_vectors, normalize_skill

# def fuzzy_match_skill(user_skill, vectorizer, skill_vectors, all_skills, threshold=0.35):
#     user_vec = vectorizer.transform([user_skill])
#     sims = cosine_similarity(user_vec, skill_vectors).flatten()
#     best_idx = sims.argmax()
#     return all_skills[best_idx] if sims[best_idx] >= threshold else None

# def predict_skills(user_skills_input, target_role, role_profiles, vectorizer, skill_vectors, all_skills, normalize_skill, top_n=20):
#     raw_user_skills = [s.strip() for s in user_skills_input.split(',') if s.strip()]
#     user_skills_normalized = [normalize_skill(s) for s in raw_user_skills]

#     user_skills_mapped = {}
#     for orig, norm in zip(raw_user_skills, user_skills_normalized):
#         matched = fuzzy_match_skill(norm, vectorizer, skill_vectors, all_skills)
#         user_skills_mapped[orig] = matched

#     mapped_set = {v for v in user_skills_mapped.values() if v is not None}

#     profile = role_profiles[target_role]
#     required_skills = profile['skills'][:top_n]
#     weights = profile['weights']

#     relevant, irrelevant = [], []
#     for orig, mapped in user_skills_mapped.items():
#         if mapped and any(
#             mapped in req or req in mapped or
#             cosine_similarity(vectorizer.transform([mapped]), vectorizer.transform([req]))[0][0] > 0.5
#             for req in required_skills
#         ):
#             relevant.append(orig)
#         else:
#             irrelevant.append(orig)

#     skills_to_learn = []
#     for req in required_skills:
#         covered = any(
#             req in m or m in req or
#             cosine_similarity(vectorizer.transform([req]), vectorizer.transform([m]))[0][0] > 0.5
#             for m in mapped_set if m
#         )
#         if not covered:
#             skills_to_learn.append(req)

#     matched_weight = sum(weights.get(req, 0) for req in required_skills if req not in skills_to_learn)
#     total_weight = sum(weights.values())
#     score = round((matched_weight / total_weight) * 100, 1) if total_weight > 0 else 0

#     return {
#         'role': target_role,
#         'match_score': score,
#         'relevant_skills': relevant,
#         'irrelevant_skills': irrelevant,
#         'skills_to_learn': skills_to_learn
#     }

# def plot_skill_gap(result):
#     categories = ['Relevant\n(Have)', 'Irrelevant\n(Have, Not Needed)', 'To Learn\n(Missing)']
#     counts = [len(result['relevant_skills']), len(result['irrelevant_skills']), len(result['skills_to_learn'])]
#     colors = ['#2ecc71', '#e74c3c', '#3498db']

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     axes[0].bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
#     axes[0].set_title(f"Skill Gap Analysis\n{result['role']}", fontsize=13, fontweight='bold')
#     axes[0].set_ylabel('Number of Skills')
#     for i, v in enumerate(counts):
#         axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

#     non_zero = [(c, cat, col) for c, cat, col in zip(counts, categories, colors) if c > 0]
#     if non_zero:
#         vals, cats, cols = zip(*non_zero)
#         axes[1].pie(vals, labels=cats, colors=cols, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
#         axes[1].set_title(f"Match Score: {result['match_score']}%", fontsize=13, fontweight='bold')

#     plt.tight_layout()
#     return fig

# # ── UI ─────────────────────────────────────────────────────────────────────────
# st.title("🎯 Intelligent Talent Match & Predictive Skill Analytics")
# st.markdown("Analyze your skill gap and discover the best-fit roles for your career.")

# df = load_data()
# unique_roles, role_profiles, all_skills, vectorizer, skill_vectors, normalize_skill = build_models(df)

# with st.sidebar:
#     st.header("👤 Your Profile")
#     current_role = st.text_input("Current Role", placeholder="e.g. Software Engineer")
#     current_level = st.selectbox(
#         "Current Level",
#         ["", "Intern / Trainee", "Fresher / Entry Level", "Junior", "Mid-Level", "Senior", "Lead", "Principal / Architect", "Manager / Director"]
#     )
#     st.markdown("---")
#     st.header("🔍 Skill Analysis")
#     user_skills = st.text_area(
#         "Your Skills (comma-separated)",
#         placeholder="e.g. Python, SQL, TensorFlow, Pandas"
#     )
#     target_role = st.selectbox("Target Role", [""] + list(unique_roles))
#     analyze_btn = st.button("Analyze Skills", type="primary", use_container_width=True)

#     st.markdown("---")
#     st.header("🌟 Discover Best Roles")
#     top_k = st.slider("Top N roles to show", 3, 10, 5)
#     discover_btn = st.button("Find Best Roles", use_container_width=True)

# # ── Profile summary ────────────────────────────────────────────────────────────
# if current_role or current_level:
#     st.subheader("📋 Your Current Profile")
#     col1, col2 = st.columns(2)
#     col1.metric("Current Role", current_role if current_role else "Not specified")
#     col2.metric("Current Level", current_level if current_level else "Not specified")
#     st.markdown("---")

# # ── Skill gap analysis ─────────────────────────────────────────────────────────
# if analyze_btn:
#     if not user_skills.strip():
#         st.warning("Please enter your skills.")
#     elif not target_role:
#         st.warning("Please select a target role.")
#     else:
#         with st.spinner("Analyzing your skills..."):
#             result = predict_skills(user_skills, target_role, role_profiles, vectorizer, skill_vectors, all_skills, normalize_skill)

#         st.subheader(f"📊 Skill Gap Analysis — {result['role']}")
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("Match Score", f"{result['match_score']}%")
#         c2.metric("✅ Relevant Skills", len(result['relevant_skills']))
#         c3.metric("❌ Irrelevant Skills", len(result['irrelevant_skills']))
#         c4.metric("📚 Skills to Learn", len(result['skills_to_learn']))

#         st.pyplot(plot_skill_gap(result))

#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.markdown("**✅ Relevant Skills** *(you have these)*")
#             for s in result['relevant_skills']:
#                 st.markdown(f"- {s}")
#         with col2:
#             st.markdown("**❌ Irrelevant Skills** *(not needed)*")
#             for s in result['irrelevant_skills']:
#                 st.markdown(f"- {s}")
#         with col3:
#             st.markdown("**📚 Skills to Learn** *(missing)*")
#             for s in result['skills_to_learn']:
#                 st.markdown(f"- {s}")

# # ── Role discovery ─────────────────────────────────────────────────────────────
# if discover_btn:
#     if not user_skills.strip():
#         st.warning("Please enter your skills first.")
#     else:
#         with st.spinner("Finding best-fit roles..."):
#             results = []
#             for role in unique_roles:
#                 try:
#                     pred = predict_skills(user_skills, role, role_profiles, vectorizer, skill_vectors, all_skills, normalize_skill)
#                     results.append({
#                         'Role': pred['role'],
#                         'Match Score (%)': pred['match_score'],
#                         'Relevant Skills': len(pred['relevant_skills']),
#                         'Skills to Learn': len(pred['skills_to_learn'])
#                     })
#                 except Exception:
#                     continue
#             result_df = pd.DataFrame(results).sort_values('Match Score (%)', ascending=False).reset_index(drop=True).head(top_k)

#         st.subheader("🌟 Best-Fit Roles for Your Skills")
#         st.dataframe(result_df, use_container_width=True)




import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import io

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="SkillSync AI Platform", page_icon="🚀", layout="wide")

# 2. Helper Functions & ML Pipeline (Cached for performance)
def normalize_skill(skill):
    """Cleans and standardizes skill text."""
    skill = skill.lower()
    skill = re.sub(r'\b(basics|fundamentals|advanced|expert|intro|introductory)\b', '', skill)
    return skill.strip()

@st.cache_data
def load_and_prepare_data():
    """Loads dataset and prepares the TF-IDF Vectorizer and Skill Matrix."""
    try:
        with open("job_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        st.error("Error: 'job_dataset.json' not found. Please ensure it's in the same directory.")
        return [], {}, None, None, []

    unique_roles = sorted({job.get("Title") for job in data if job.get("Title") is not None})
    
    # Build Role Profiles (Aggregated skills per role)
    role_profiles = {}
    for role in unique_roles:
        role_jobs = [j for j in data if j.get("Title") == role]
        skills = []
        for j in role_jobs:
            skills.extend([normalize_skill(s) for s in j.get("Skills", [])])
        
        # Keep top 20 most frequent skills for the industry standard
        skill_counts = Counter(skills)
        top_skills = [s[0] for s in skill_counts.most_common(20)]
        role_profiles[role] = top_skills

    # Build unique vocabulary for the vectorizer
    all_skills = set()
    for skills in role_profiles.values():
        all_skills.update(skills)
    all_skills = sorted(list(all_skills))

    # Train TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(all_skills)
    
    # Pre-compute skill vectors for fast similarity checks
    skill_vectors = {skill: vectorizer.transform([skill]) for skill in all_skills}

    return unique_roles, role_profiles, vectorizer, skill_vectors, all_skills

def predict_skills(user_skills_str, target_role, role_profiles, vectorizer, skill_vectors, all_skills):
    """Core ML prediction logic using Cosine Similarity."""
    target_skills = role_profiles.get(target_role, [])
    if not target_skills:
        return None
       
    user_skills_list = [normalize_skill(s) for s in user_skills_str.split(",") if s.strip()]
    if not user_skills_list:
        return None

    user_vecs = vectorizer.transform(user_skills_list)
    
    relevant = set()
    irrelevant = set(user_skills_list)
    
    # Calculate similarity thresholds
    for i, u_skill in enumerate(user_skills_list):
        u_vec = user_vecs[i]
        for t_skill in target_skills:
            t_vec = skill_vectors[t_skill]
            sim = cosine_similarity(u_vec, t_vec)[0][0]
            if sim > 0.6:  # Threshold for semantic match
                relevant.add(u_skill)
                if u_skill in irrelevant:
                    irrelevant.remove(u_skill)
                break

    skills_to_learn = [s for s in target_skills if not any(cosine_similarity(skill_vectors[s], vectorizer.transform([u]))[0][0] > 0.6 for u in user_skills_list)]
    
    match_score = min(max(round((len(relevant) / len(target_skills)) * 100), 5), 100) if target_skills else 0
    
    return {
        "role": target_role,
        "match_score": match_score,
        "relevant_skills": list(relevant),
        "irrelevant_skills": list(irrelevant),
        "skills_to_learn": skills_to_learn
    }

def plot_skill_gap(result):
    """Generate a matplotlib figure showing skill gap analysis."""
    categories = ['Relevant\n(Have)', 'Irrelevant\n(Have, Not Needed)', 'To Learn\n(Missing)']
    counts = [len(result['relevant_skills']), len(result['irrelevant_skills']), len(result['skills_to_learn'])]
    colors = ['#2ecc71', '#e74c3c', '#3498db']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
    axes[0].set_title(f"Skill Gap Analysis\n{result['role']}", fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Number of Skills')
    for i, v in enumerate(counts):
        axes[0].text(i, v + 0.1, str(v), ha='center', fontweight='bold')

    non_zero = [(c, cat, col) for c, cat, col in zip(counts, categories, colors) if c > 0]
    if non_zero:
        vals, cats, cols = zip(*non_zero)
        axes[1].pie(vals, labels=cats, colors=cols, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 10})
        axes[1].set_title(f"Match Score: {result['match_score']}%", fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig

# 3. Load Data
unique_roles, role_profiles, vectorizer, skill_vectors, all_skills = load_and_prepare_data()

# Sidebar for global inputs
with st.sidebar:
    st.header("👤 Your Profile")
    current_role = st.text_input("Current Role", placeholder="e.g. Software Engineer")
    current_level = st.selectbox(
        "Current Level",
        ["", "Intern / Trainee", "Fresher / Entry Level", "Junior", "Mid-Level", "Senior", "Lead", "Principal / Architect", "Manager / Director"]
    )
    st.markdown("---")
    user_skills_input = st.text_area(
        "Enter your current skills (comma separated):", 
        value="python, pandas, basic sql, communication",
        height=150,
        help="E.g. Python, React, Data Analysis, SQL"
    )
    st.markdown("---")
    st.markdown("💡 *Powered by TF-IDF & Cosine Similarity*")

# Profile Summary
if current_role or current_level:
    st.subheader("📋 Your Current Profile")
    col1, col2 = st.columns(2)
    col1.metric("Current Role", current_role if current_role else "Not specified")
    col2.metric("Current Level", current_level if current_level else "Not specified")
    st.divider()

# 4. Streamlit UI Layout
st.title("🚀 SkillSync AI")
st.markdown("### AI-Powered Career Intelligence & Gap Analysis")
st.divider()

# Create Tabs for a cleaner layout
tab1, tab2 = st.tabs(["🎯 Target Role Analyzer", "🧭 AI Role Discovery"])

# ── TAB 1: GAP ANALYZER ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Analyze your fit for a specific role")
    colA, colB = st.columns([3, 1])
    with colA:
        target_role_input = st.selectbox("Select Target Role:", options=unique_roles)
    with colB:
        st.write("") # Spacing
        st.write("") # Spacing
        analyze_btn = st.button("Run Gap Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_skills_input.strip():
            st.warning("⚠️ Please enter your skills in the sidebar first.")
        else:
            with st.spinner("Analyzing skill vectors..."):
                result = predict_skills(user_skills_input, target_role_input, role_profiles, vectorizer, skill_vectors, all_skills)
            
            if result:
                # Top Metrics Row
                m1, m2, m3 = st.columns(3)
                m1.metric("Match Score", f"{result['match_score']}%", "Placement Probability")
                m2.metric("Verified Skills", len(result['relevant_skills']))
                m3.metric("Skills to Learn", len(result['skills_to_learn']), delta="-Gap", delta_color="inverse")
                
                st.markdown("---")
                
                # Detailed Breakdown Columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                with res_col1:
                    st.success("✅ Relevant Skills (Matched)")
                    for s in result['relevant_skills']:
                        st.markdown(f"- {s.title()}")
                         
                with res_col2:
                    st.warning("⚠️ Irrelevant Skills (Not Core)")
                    for s in result['irrelevant_skills']:
                        st.markdown(f"- {s.title()}")
                         
                with res_col3:
                    st.error("📚 Action Plan (Skills to Learn)")
                    for s in result['skills_to_learn']:
                        st.markdown(f"- **{s.title()}**")
                
                # Skill Gap Plot
                st.markdown("---")
                st.subheader("Skill Gap Visualization")
                fig = plot_skill_gap(result)
                st.pyplot(fig)
                
                # Export Functionality
                st.markdown("---")
                st.subheader("📥 Export Action Plan")
                
                # Create a DataFrame for the export
                export_df = pd.DataFrame({
                    "Category": ["Matched"]*len(result['relevant_skills']) + ["To Learn"]*len(result['skills_to_learn']),
                    "Skill": result['relevant_skills'] + result['skills_to_learn']
                })
                csv = export_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Gap Report (CSV)",
                    data=csv,
                    file_name=f"{target_role_input.replace(' ', '_')}_Gap_Report.csv",
                    mime="text/csv",
                )

# ── TAB 2: ROLE DISCOVERY ────────────────────────────────────────────────────
with tab2:
    st.subheader("Discover the best roles for your current skills")
    
    top_k_input = st.slider("Number of roles to recommend:", min_value=1, max_value=10, value=3)
    discover_btn = st.button("Find Best-Fit Roles", type="primary")
    
    if discover_btn:
        if not user_skills_input.strip():
            st.warning("⚠️ Please enter your skills in the sidebar first.")
        else:
            with st.spinner("Scanning industry dataset..."):
                results = []
                for role in unique_roles:
                    pred = predict_skills(user_skills_input, role, role_profiles, vectorizer, skill_vectors, all_skills)
                    if pred and pred['match_score'] > 0:
                        results.append({
                            'Role': pred['role'],
                            'Match Score (%)': pred['match_score'],
                            'Matched Skills Count': len(pred['relevant_skills']),
                            'Skills to Learn Count': len(pred['skills_to_learn'])
                        })
                
                if results:
                    result_df = pd.DataFrame(results).sort_values('Match Score (%)', ascending=False).reset_index(drop=True).head(top_k_input)
                    
                    st.dataframe(
                        result_df.style.background_gradient(cmap='Blues', subset=['Match Score (%)']), 
                        use_container_width=True
                    )
                    
                    # Export Discovery Results
                    csv_discovery = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Recommendations (CSV)",
                        data=csv_discovery,
                        file_name="AI_Role_Recommendations.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No strong matches found. Try adding more general skills to your profile.")