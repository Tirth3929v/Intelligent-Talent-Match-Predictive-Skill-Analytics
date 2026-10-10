import unittest
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest


class StreamlitAnalyzerTests(unittest.TestCase):
    def test_gap_analyzer_posts_profile_and_renders_agent_response(self) -> None:
        payload = {
            "match_score_percentage": 77,
            "market_benchmarks_summary": "Current market summary.",
            "skill_breakdown": {
                "relevant_matched": ["Python"],
                "transferable_or_secondary": ["SQL"],
                "missing_critical": ["MLOps"],
            },
            "step_by_step_learning_roadmap": [
                {
                    "phase": "Build",
                    "skills": ["MLOps"],
                    "recommended_project": "Deploy a model.",
                }
            ],
            "interview_prep_questions": ["How do you monitor a model?"],
            "search_sources": ["https://example.com/jobs"],
        }
        response = Mock(status_code=200)
        response.json.return_value = payload

        with patch("requests.post", return_value=response) as post:
            app = AppTest.from_file("app.py").run(timeout=30)
            app.text_input[0].set_value("Data Analyst")
            app.selectbox[1].select("Junior")
            app.button(key="btn_run_gap").click().run(timeout=30)

        self.assertFalse(app.exception)
        self.assertEqual(post.call_args.kwargs["json"], {
            "current_role": "Data Analyst",
            "current_level": "Junior",
            "user_skills": ["python", "pandas", "basic sql", "communication"],
            "target_role": ".NET Developer",
        })
        self.assertTrue(any(item.value == "77%" for item in app.metric))
        self.assertTrue(
            any("Current market summary." in item.value for item in app.info)
        )
        self.assertTrue(
            any("Recommended project" in item.value for item in app.info)
        )


if __name__ == "__main__":
    unittest.main()
