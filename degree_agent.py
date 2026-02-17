import json
import os
from typing import Optional
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ.get("Gemini_API_Key"))
MODEL_ID = "gemini-2.5-flash-lite"
KB_JSON_PATH = "knowledge_base/degree_planning.json"

TAG_EXTRACTION_PROMPT = """
You are an academic advisor assistant for Cal Poly Computer Science MS and BMS students.

A student has asked a question. Your job is to identify which topic tags
from the AVAILABLE TAGS list best describe what the student is asking about.

Rules:
- ONLY choose tags from the AVAILABLE TAGS list below — never invent new ones
- Return between 2 and 7 tags that best match the topic
- Return ONLY valid JSON — no extra text, no explanation
- Format: {{"selected_tags": ["tag1", "tag2"]}}

Student's question:
{query}

AVAILABLE TAGS:
{available_tags}
"""


DEGREE_PLANNING_PROMPT = """
You are an academic advisor assistant for Cal Poly Computer Science graduate students.
You help students in the MS and Blended B.S. + M.S. (BMS) programs understand
their degree requirements, plan their coursework, and navigate program policies.

IMPORTANT RULES:
- Answer ONLY using the KNOWLEDGE BASE provided below
- Never invent policies, deadlines, unit counts, or course numbers
- Always cite the source URL for the information you use
- If the answer is not in the knowledge base, say so and direct the student
  to contact the graduate coordinator at bellardo@calpoly.edu
- Be friendly, clear, and specific — students are often stressed about deadlines
- When relevant, flag important warnings (e.g., GWR must be complete BEFORE finishing BS)

Student's question:
{query}

KNOWLEDGE BASE (use only this information to answer):
{knowledge_context}
"""


def load_knowledge_base(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # The JSON has a top-level "chunks" key containing the list we want.
    return data.get("chunks", [])


def extract_all_tags(chunks: list[dict]) -> list[str]:
    tag_set = set()
    for chunk in chunks:
        for tag in chunk.get("tags", []):
            tag_set.add(tag.lower())
    return sorted(tag_set)


def extract_relevant_tags(query: str, all_tags: list[str]) -> list[str]:
    prompt = TAG_EXTRACTION_PROMPT.format(
        query=query,
        available_tags=", ".join(all_tags)
    )

    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    raw = response.text.strip()

    raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(raw)
    return parsed.get("selected_tags", [])


def filter_chunks(
    chunks: list[dict],
    selected_tags: list[str],
    limit: int = 8
) -> list[dict]:
    if not selected_tags:
        return chunks[:limit]

    selected_lower = [t.lower() for t in selected_tags]

    scored = []
    for chunk in chunks:
        chunk_tags = [t.lower() for t in chunk.get("tags", [])]
        # Count how many selected tags appear in this chunk's tags.
        score = sum(1 for t in selected_lower if t in chunk_tags)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in scored[:limit]]


def build_knowledge_context(chunks: list[dict]) -> str:
    if not chunks:
        return (
            "NO RELEVANT PROGRAM INFORMATION FOUND. "
            "Direct the student to contact bellardo@calpoly.edu."
        )

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] {chunk['title']}\n"
            f"    Info: {chunk['summary']}\n"
            f"    Source: {chunk['sourceURL']}"
        )

    return "\n\n".join(blocks)


def answer_student_query(query: str, knowledge_context: str) -> str:
    prompt = DEGREE_PLANNING_PROMPT.format(
        query=query,
        knowledge_context=knowledge_context
    )

    response = client.models.generate_content(model=MODEL_ID, contents=prompt)
    return response.text


def run_degree_planning_agent(query: str) -> str:
    # Step 1: Load all knowledge chunks from the JSON file.
    chunks = load_knowledge_base(KB_JSON_PATH)

    # Step 2: Get every tag that exists across all chunks.
    all_tags = extract_all_tags(chunks)

    # Step 3: Use the AI to identify which tags are relevant to this question.
    print(f"[Agent] Identifying relevant topics for: '{query}'")
    selected_tags = extract_relevant_tags(query, all_tags)
    print(f"[Agent] Selected tags: {selected_tags}")

    # Step 4: Keep only the chunks that match those tags.
    relevant_chunks = filter_chunks(chunks, selected_tags)
    print(f"[Agent] Retrieved {len(relevant_chunks)} relevant knowledge chunks.")

    # Step 5: Format the chunks into a readable context block.
    context = build_knowledge_context(relevant_chunks)

    # Step 6: Get the AI's answer using the student's question + context.
    answer = answer_student_query(query, context)

    return answer
