import gradio as gr
import json
import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

endpoint = "https://gradgpt-openai.openai.azure.com/"
deployment = "gradgpt-chat"
api_version = "2024-12-01-preview"
subscription_key = os.environ.get("Azure_API_Key")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

KB_JSON_PATH = "knowledge_base/deadlines.json"

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


def load_knowledge_base(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("forms", [])


def extract_all_tags(forms):
    tag_set = set()
    for form in forms:
        for tag in form.get("tags", []):
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

def filter_forms(forms, selected_tags, limit=8):
    if not selected_tags:
        return forms[:limit]

    selected_lower = [t.lower() for t in selected_tags]
    scored = []

    for form in forms:
        form_tags = [t.lower() for t in form.get("tags", [])]
        score = sum(1 for t in selected_lower if t in form_tags)
        if score > 0:
            scored.append((score, form))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [form for _, form in scored[:limit]]


# Converting forms into easier to read sections  
def build_knowledge_context(forms):
    if not forms:
        return "NO RELEVANT INFO FOUND. Contact bellardo@calpoly.edu"

    blocks = []
    for i, form in enumerate(forms, 1):
        blocks.append(
            f"[{i}] {form['title']}\n"
            f"Description: {form['content']}\n"
            f"Deadline: {form.get('deadline', 'Not specified')}\n"
            f"Term: {form.get('term', 'Not specified')}\n"
            f"Applies To: {', '.join(form.get('appliesTo', []))}\n"
            f"Official URL: {form['sourceURL']}"
        )

    return "\n\n".join(blocks)


def answer_student_query(query, knowledge_context):
    user_prompt = f"""
        Student Question:
        {query}

        KNOWLEDGE BASE:
        {knowledge_context}
    """

    return azure_chat( # Officially calling AI endpoint
        system_message=DEADLINES_PROMPT,
        user_message=user_prompt
    )

# def answer_student_query(query, knowledge_context):
#     formatted_system_prompt = DEADLINES_PROMPT.format(
#         query=query,
#         knowledge_context=knowledge_context
#     )

#     return azure_chat(
#         system_message=formatted_system_prompt,
#         user_message="Answer the student's question."
#     )


def run_forms_and_deadlines_agent(query):
    # Step 1: Load all knowledge forms from the JSON file.
    forms = load_knowledge_base(KB_JSON_PATH)

    # Step 2: Get every tag that exists across all forms.
    all_tags = extract_all_tags(forms)

    # Step 3: Use the AI to identify which tags are relevant to this question.
    print(f"[Agent] Identifying relevant topics for: '{query}'")
    selected_tags = extract_relevant_tags(query, all_tags)
    print(f"[Agent] Selected tags: {selected_tags}")

    # Step 4: Keep only the forms that match those tags.
    relevant_forms = filter_forms(forms, selected_tags)
    print(f"[Agent] Retrieved {len(relevant_forms)} relevant knowledge forms.")

    # Step 5: Format the forms into a readable context block.
    context = build_knowledge_context(relevant_forms)

    # Step 6: Get the AI's answer using the student's question + context.
    answer = answer_student_query(query, context)

    return answer


def chat_with_gpt(message, history):
    if history is None:
        history = []

    history.append(gr.ChatMessage(role="user", content=message))

    try:
        response = run_forms_and_deadlines_agent(message)
    except Exception as e:
        response = f"Error: {str(e)}"

    history.append(gr.ChatMessage(role="assistant", content=response))

    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Grad-GPT (Azure Version)")
    gr.Markdown("Forms and Deadlines RAG Agent using Azure OpenAI")

    chatbot = gr.Chatbot(height=400)
    chat_input = gr.Textbox(placeholder="Ask your forms and deadlines question...")
    send_button = gr.Button("Send")

    send_button.click(chat_with_gpt, [chat_input, chatbot], [chatbot, chat_input])
    chat_input.submit(chat_with_gpt, [chat_input, chatbot], [chatbot, chat_input])

demo.launch()