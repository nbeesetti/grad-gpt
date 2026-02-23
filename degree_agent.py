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


DEGREE_PLANNING_PROMPT = """
You are an academic advisor assistant for Cal Poly Computer Science MS and BMS students.
IMPORTANT RULES:
- Answer ONLY using the KNOWLEDGE BASE provided
- Never invent policies, deadlines, unit counts, or course numbers
- Always cite source URLs
- If information is not available, instruct the student to contact bellardo@calpoly.edu.
- Be friendly, clear, and concise
- When relevant, flag important warnings (e.g., GWR must be complete BEFORE finishing BS)
Student's question:
{query}
KNOWLEDGE BASE:
{knowledge_context}
"""

def load_knowledge_base_from_supabase():
    response = (
        supabase
        .table("KnowledgeBase")
        .select("*")
        .contains("agentIds", ["3"])  # 3 = "degree progress agent"
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
        max_completion_tokens=1024,
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
    system_prompt = DEGREE_PLANNING_PROMPT.format(
        query=query,
        knowledge_context=knowledge_context
    )

    return azure_chat(
        system_message=system_prompt,
        user_message="Please answer based on the knowledge base provided above."
    )


def run_degree_planning_agent(query):
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

def chat_with_gpt(message, history):
    if history is None:
        history = []

    history.append(gr.ChatMessage(role="user", content=message))

    try:
        response = run_degree_planning_agent(message)
    except Exception as e:
        response = f"Error: {str(e)}"

    history.append(gr.ChatMessage(role="assistant", content=response))

    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Grad-GPT (Azure Version)")
    gr.Markdown("Degree Planning RAG Agent using Azure OpenAI")

    chatbot = gr.Chatbot(height=400)
    chat_input = gr.Textbox(placeholder="Ask your degree planning question...")
    send_button = gr.Button("Send")

    send_button.click(chat_with_gpt, [chat_input, chatbot], [chatbot, chat_input])
    chat_input.submit(chat_with_gpt, [chat_input, chatbot], [chatbot, chat_input])

demo.launch()