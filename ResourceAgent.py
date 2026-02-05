import json
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv

# from googlegenai import AzureOpenAI
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("Gemini_API_Key"))
MODEL_ID = "gemini-2.5-flash-lite"


def generate_text(prompt):
    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text


SYSTEM_PROMPT = """
You are a research support assistant for CS Master's students.

You may ONLY recommend resources that are provided.
You must NEVER invent links, tools, or websites.
If no relevant resources are available, say so clearly.

Your task:
- Select the most relevant resources
- Briefly explain why each is useful
- Return at most 5 resources
- Always include the link exactly as provided
"""

USER_PROMPT_TEMPLATE = """
The student is working on: {topic}

AVAILABLE RESOURCES:
{resource_context}

Respond with a concise, helpful list.
"""


def load_resources(path):
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
        if category:
            if r.get("category", "").lower() != category.lower():
                continue

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
            Category: {r["category"]}
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
    resources = load_resources("knowledge_base/resources.json")

    topic = "literature review"

    filtered = filter_resources(
        resources,
        category="Literature Review",
        tags=["lit", "research"],
    )

    context = build_resource_context(filtered)

    answer = run_resource_agent(
        topic=topic,
        resource_context=context,
    )

    print(answer)


if __name__ == "__main__":
    main()
