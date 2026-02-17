import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# from googlegenai import AzureOpenAI
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("Gemini_API_Key"))
MODEL_ID = "gemini-2.5-flash-lite"
RES_JSON_PATH = "knowledge_base/resources.json"

TAG_EXTRACTION_PROMPT = """
You are a research support assistant for Computer Science Master's students.

From the student's query, select the most relevant tags
from the AVAILABLE TAGS list.

Rules:
- Only choose tags from AVAILABLE TAGS
- Return 2-7 tags max
- Return valid JSON response only
- Format: {"selected_tags": ["tag1", "tag2"]}

Student's query:
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

# -----------------------------------------------------------------------------


def generate_text(prompt):
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text


def load_resources_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("resources", [])


def filter_resources(
    resources,
    category,
    tags,
    limit=5,
):
    results = []

    for r in resources:
        if tags:
            resource_tags = [t.lower() for t in r.get("tags", [])]
            if not any(tag.lower() in resource_tags for tag in tags):
                continue

        results.append(r)

        if len(results) >= limit:
            break

    return results


def build_resource_context(resources):
    if not resources:
        return "NO RESOURCES AVAILABLE."

    blocks = []
    for i, r in enumerate(resources, start=1):
        blocks.append(
            f"""
            [{i}
            Title: {r["title"]}
            Description: {r["description"]}
            Link: {r["url"]}"""
        )

    return "\n\n".join(blocks)


def build_full_prompt(topic, resource_context):
    user_prompt = USER_PROMPT_TEMPLATE.format(
        topic=topic,
        resource_context=resource_context,
    )
    return SYSTEM_PROMPT.strip() + "\n\n" + user_prompt.strip()


def run_resource_agent(topic, resource_context):
    prompt = build_full_prompt(topic, resource_context)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    return response.text


def main():
    resources_json = load_resources_json(RES_JSON_PATH)

    tag = "literature review"

    filtered = filter_resources(
        resources_json,
        tags=["lit-review", "research"],
    )

    context = build_resource_context(filtered)

    answer = run_resource_agent(
        topic=topic,
        resource_context=context,
    )

    print(answer)


if __name__ == "__main__":
    main()
