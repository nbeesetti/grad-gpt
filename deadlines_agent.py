import gradio as gr
import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from supabase import create_client, Client


load_dotenv()

endpoint = "https://gradgpt-openai.openai.azure.com/"
deployment = "gradgpt-chat"
api_version = "2024-12-01-preview"
subscription_key = os.environ.get("Azure_API_Key")

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(supabase_url, supabase_key)

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

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


DEADLINES_PROMPT = """
You are an administrative assistant for Cal Poly Computer Science graduate students.
You help students in the MS and Blended B.S. + M.S. (BMS) programs understand required forms, 
administrative deadlines, and graduation-related paperwork based on their academic term and 
progress.

IMPORTANT RULES:
- Answer ONLY using the KNOWLEDGE BASE provided below
- Never invent forms, deadlines, policies, or submission requirements.
- Only reference forms that exist in the knowledge base.
- Always include the form’s official URL when referencing it.
- If a requested form or deadline is not in the knowledge base, say so clearly and direct 
the student to contact the graduate coordinator at bellardo@calpoly.edu.
- Be friendly, clear, and specific — students are often stressed about deadlines.
- When discussing deadlines, always include: The exact due date, The term it applies to
Whether it applies to CSC-MS, CSC-BMS, or both (based on appliesTo)
- If multiple forms apply, list them clearly in bullet points.
- If a deadline has passed, clearly warn the student.

Student's question:
{query}

KNOWLEDGE BASE (use only this information to answer):
{knowledge_context}
"""


def load_knowledge_base_from_supabase():
    response = (
        supabase
        .table("KnowledgeBase")
        .select("*")
        .contains("agentIds", ["4"])  # 4 = "forms and deadlines agent"
        .execute()
    )

    if response.data is None:
        return []

    return response.data


def extract_all_tags(chunks):
    tag_set = set()
    for chunk in chunks:
        for tag in chunk.get("tags", []):
            tag_set.add(tag.lower())
    return sorted(tag_set)


def azure_chat(system_message, user_message):
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        # changed from 1024 to 2048 for better handling of longer contexts
        max_completion_tokens=2048,
    )

    return response.choices[0].message.content


def extract_relevant_tags(query, all_tags):
    prompt = TAG_EXTRACTION_PROMPT.format(
        query=query,
        available_tags=", ".join(all_tags)
    )

    raw = azure_chat(
        system_message="You extract structured JSON only.",
        user_message=prompt
    )

    raw = raw.strip().replace("```json", "").replace("```", "").strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("selected_tags", [])
    except:
        return []


def filter_chunks(chunks, selected_tags, limit=8):
    if not selected_tags:
        return chunks[:limit]

    selected_lower = [t.lower() for t in selected_tags]
    scored = []

    for chunk in chunks:
        chunk_tags = [t.lower() for t in chunk.get("tags", [])]
        score = sum(1 for t in selected_lower if t in chunk_tags)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]


# Converting chunks into easier to read sections
def build_knowledge_context(chunks, max_chars=4000):
    if not chunks:
        return "NO RELEVANT INFO FOUND. Contact bellardo@calpoly.edu"

    blocks = []
    total_len = 0

    for i, chunk in enumerate(chunks, 1):
        block = f"[{i}] {chunk['title']}\nInfo: {chunk['content']}\nSource: {chunk['sourceURL']}\n\n"
        if total_len + len(block) > max_chars:
            break
        blocks.append(block)
        total_len += len(block)

    return "".join(blocks)


def answer_student_query(query, knowledge_context):
    system_prompt = DEADLINES_PROMPT.format(
        query=query,
        knowledge_context=knowledge_context
    )

    return azure_chat(
        system_message=system_prompt,
        user_message="Please answer based on the knowledge base provided above."
    )


def run_forms_and_deadlines_agent(query):
    chunks = load_knowledge_base_from_supabase()
    print(f"[DEBUG] Loaded {len(chunks)} chunks from Supabase")

    all_tags = extract_all_tags(chunks)
    print(f"[DEBUG] Total unique tags: {len(all_tags)}")

    selected_tags = extract_relevant_tags(query, all_tags)
    print(f"[DEBUG] Selected tags: {selected_tags}")

    relevant_chunks = filter_chunks(chunks, selected_tags)
    print(f"[DEBUG] Retrieved {len(relevant_chunks)} relevant chunks")

    context = build_knowledge_context(relevant_chunks)
    print(f"[DEBUG] Context length: {len(context)} characters")

    answer = answer_student_query(query, context)
    print(f"[DEBUG] Azure response: {answer}")

    return answer
