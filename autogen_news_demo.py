# autogen_news_demo_ollama_fixed.py
"""
AutoGen Multi-Agent News Demo with Ollama codellama:7b-instruct
- Reporter Agent: fetches simulated "live" data internally
- Editorial Agent: writes newsletter intro using LLM
- Sequential pipeline: Reporter -> Editorial
"""
from autogen import AssistantAgent
import ollama  # Make sure Ollama is installed and codellama:7b-instruct is pulled
import random

MODEL_NAME = "codellama:7b-instruct"

# ---------- Agents ----------
class ReporterAgent(AssistantAgent):
    """
    Specialized agent to collect news data (simulated)
    """
    def respond(self, message: str) -> str:
        topic = message
        # Simulate fetching 3 "live" news items
        news_items = [
            f"{topic} news update #{i}: {random.choice(['Market rises', 'Market falls', 'AI breakthrough', 'Tech trend'])}"
            for i in range(1, 4)
        ]
        print(f"[Reporter] Fetched news for topic '{topic}': {news_items}")
        return "\n".join(news_items)

class EditorialAgent(AssistantAgent):
    """
    Specialized agent to write newsletter intro using Ollama LLM
    """
    def respond(self, message: str) -> str:
        """
        message: raw news lines
        """
        prompt = (
            "You are an editor writing a newsletter intro. "
            "Use the following raw news items to create a concise and engaging introduction:\n"
            f"{message}"
        )
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

# ---------- Run Sequential Multi-Agent Pipeline ----------
def run_pipeline(topic: str):
    # Initialize Agents
    reporter = ReporterAgent(name="Reporter")
    editorial = EditorialAgent(name="Editorial")

    # Sequential flow: User -> Reporter -> Editorial
    print(f"=== Topic: {topic} ===\n")
    raw_news = reporter.respond(topic)
    print(f"[Reporter Output]\n{raw_news}\n")

    final_newsletter = editorial.respond(raw_news)
    print(f"[Editorial Output]\n{final_newsletter}\n")

if __name__ == "__main__":
    # Example run
    run_pipeline("AI Stock Trends")
