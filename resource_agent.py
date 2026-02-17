import json
import os
from typing import List, Dict
from dotenv import load_dotenv

# from googlegenai import AzureOpenAI
from google import genai

load_dotenv()

DEBUG = False
MODEL_ID = "gemini-2.5-flash-lite"
RES_JSON_PATH = "knowledge_base/resources.json"


# Prompts -----------------------------------------------------------------


TAG_EXTRACTION_PROMPT = """
You are a research support assistant for Computer Science Master's students.

From the student's query, select the most relevant tags
from the AVAILABLE TAGS list.

Rules:
- Only choose tags from AVAILABLE TAGS
- Return 2-7 tags max
- Return valid JSON response only
- Response Format: {{"selected_tags": ["tag1", "tag2"]}}

Student query:
{query}

AVAILABLE TAGS:
{available_tags}
"""


RESOURCE_RANKING_PROMPT = """
You are a research support assistant for Computer Science Master's students.

You may ONLY recommend resources provided below.
You must NEVER invent links or tools.

Your task:
- Select the most relevant resources
- Rank them by relevance to student's query
- Return at most 5
- Briefly explain why each is useful
- Always include the link EXACTLY as provided

Student query:
{query}

AVAILABLE RESOURCES:
{resource_context}
"""


# Resource Agent Class --------------------------------------------------------


class ResourceAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("Gemini_API_Key"))
        self.resources_json = self._load_resources_json(RES_JSON_PATH)
        self.available_tags = self._load_available_tags(self.resources_json)

    def _load_resources_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("resources", [])

    def _load_available_tags(self, resources: List[Dict]) -> set:
        return {tag.lower() for r in resources for tag in r.get("tags", [])}

    # Make LLM Calls ------------------------------------------------

    def _generate(self, prompt: str):
        response = self.client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        if DEBUG:
            print(response.text)
        return response.text

    # Functions -----------------------------------------------------

    def get_tags(self, query: str) -> set:
        prompt = TAG_EXTRACTION_PROMPT.format(
            query=query, available_tags=(", ".join(self.available_tags))
        )

        response = self._generate(prompt)
        print("RESPONSE IN GET TAGS:", json.loads(response))

        tags = json.loads(response).get("selected_tags", [])
        return {t.lower() for t in tags if t in self.available_tags}

    def filter_resources(self, selected_tags: set):
        res = []
        for r in self.resources_json:
            resource_tags = {t.lower() for t in r.get("tags", [])}
            if selected_tags.intersection(resource_tags):
                res.append(r)
        return res

    def rank_resources(self, query, filtered_resources: List[Dict]):
        if not filtered_resources:
            return "None found."

        resource_context = []
        for i, r in enumerate(filtered_resources):
            resource_context.append(
                f"[{i+1}]\nTitle: {r['title']}\nDescription: {r['description']}\nLink: {r['url']}\nTags: {', '.join(r.get('tags', []))}"
            )
        context = "\n\n".join(resource_context)

        prompt = RESOURCE_RANKING_PROMPT.format(query=query, resource_context=context)

        return self._generate(prompt)

    def run(self, query: str):
        selected_tags = self.get_tags(query)
        filtered_res = self.filter_resources(selected_tags)
        return self.rank_resources(query, filtered_res)


# Main ------------------------------------------------------------------------


def main():
    res_agent = ResourceAgent()

    while True:
        query = input("Student's resource-related query: ")
        agent_response = res_agent.run(query)
        print("\n", agent_response)


if __name__ == "__main__":
    main()
