import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ClaudeAssistant:
    """Interface to Claude 3 Haiku API for dental analysis assistance"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-haiku-20240307"

    def generate_dental_report(self, conditions, ontology_data):
        """Generate a detailed analysis report for detected dental conditions"""
        if not self.api_key:
            return {"error": "API key not configured. Please set the ANTHROPIC_API_KEY environment variable."}

        # Create a prompt for Claude with detected conditions and ontology data
        prompt = self._create_dental_prompt(conditions, ontology_data)

        print(f"Using API key: {self.api_key[:5]}...")

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }

            data = {
                "model": self.model,
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            response = requests.post(self.api_url, headers=headers, json=data)

            if response.status_code != 200:
                print(f"API error: {response.status_code} {response.text}")
                return {"error": f"API error: {response.status_code} {response.text}"}

            response.raise_for_status()

            result = response.json()
            content = result.get("content", [])

            if content and len(content) > 0:
                report = content[0].get("text", "")
                return {"report": report}
            else:
                return {"error": "No content in response"}

        except Exception as e:
            print(f"Error generating dental report: {str(e)}")
            return {
                "error": f"Error generating dental report: {str(e)}"
            }

    def _create_dental_prompt(self, conditions, ontology_data):
        """Create a concise prompt for Claude with dental information"""
        # Assign a number to each condition
        numbered_conditions = [f"{i + 1}. {cond['condition']} (confidence: {cond['probability']:.1%})"
                               for i, cond in enumerate(conditions)]
        conditions_text = "\n".join(numbered_conditions)

        prompt = f"""As a dental AI assistant, provide a brief analysis of this dental X-ray.
    Detected conditions:
    {conditions_text}

    For each numbered condition, give a 1-2 sentence description and a short treatment recommendation.
    Be extremely concise. Total response should be under 250 words.
    Use numbers (1, 2, 3, etc.) that correspond to the conditions above to organize your report.
    """
        return prompt