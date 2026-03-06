import gradio as gr
import json
import os
import re
import traceback
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

ROUTER_AND_FILTER_PROMPT = """
Classify the student question and extract structured course filters.

Return JSON only (no markdown, no backticks):

{{
  "intent": "COURSE_ONLY" | "KB_ONLY" | "HYBRID",
  "levels": ["400", "500"] | null,
  "topic": string | null
}}

Rules:
- Use "COURSE_ONLY" if the student ONLY asks for a list of courses or information about a specific course to take
- Use "KB_ONLY" if the student asks about policies, requirements, faculty, or procedures
- Use "HYBRID" if the student asks about courses AND policies/requirements
- levels should contain hundred-levels (e.g. ["400","500"])
- topic should be short (e.g. "Artificial Intelligence", "Machine Learning")
- If no levels mentioned, return null
- If no topic mentioned, return null

Question:
{query}
"""

KB_ANSWER_PROMPT = """
You are an academic advisor assistant for Cal Poly's graduate CS program.
Answer ONLY using the provided knowledge base entries.
If the information is missing, say "Please contact bellardo@calpoly.edu for more information."

{user_context}

KNOWLEDGE BASE:
{knowledge_context}

Question:
{query}
"""

LLM_TOPIC_FILTER_PROMPT = """
You are filtering university courses by topic relevance.

Student topic:
"{topic}"

Courses:
{courses_json}

Return JSON only (no markdown, no backticks):
{{"relevant_course_nums": ["CSC 400", "CSC 530"]}}
Be generous — include reasonably related courses.
"""

def load_knowledge_base_from_supabase():
    try:
        print("[KB] Loading knowledge base from Supabase...")
        response = (
            supabase
            .table("KnowledgeBase")
            .select("id, title, content, sourceURL, tags, agentIds")
            .contains("agentIds", ["3"])  # agentId 3 = degree planning
            .execute()
        )
        entries = response.data or []
        print(f"[KB] Loaded {len(entries)} KB entries")
        if entries:
            print(f"[KB] Sample entry titles: {[e.get('title','N/A') for e in entries[:3]]}")
        return entries
    except Exception as e:
        print(f"[KB] ERROR loading KB: {str(e)}")
        return []


KB_CACHE = load_knowledge_base_from_supabase()

def load_user_context(user_id: int):
    try:
        print(f"[USER] Loading user data for id: {user_id}")
        response = (
            supabase
            .table("Users")
            .select("id, completedCourses, currentCourses, graduationTarget, startTerm, plannedCourses")
            .eq("id", user_id)
            .single()
            .execute()
        )
        user = response.data
        print(f"[USER] Loaded user: {user}")
        return user
    except Exception as e:
        print(f"[USER] ERROR loading user: {str(e)}")
        return None

def format_user_context(user):
    if not user:
        return ""
    lines = [
        "STUDENT PROFILE:",
        f"- Start Term: {user.get('startTerm', 'N/A')}",
        f"- Graduation Target: {user.get('graduationTarget', 'N/A')}",
        f"- Completed Courses: {', '.join(user.get('completedCourses') or []) or 'None'}",
        f"- Current Courses: {', '.join(user.get('currentCourses') or []) or 'None'}",
        f"- Planned Courses: {', '.join(user.get('plannedCourses') or []) or 'None'}",
    ]
    return "\n".join(lines)

def azure_json_call(system_msg, user_msg, max_completion_tokens=2000):
    print("\n[AZURE] --- JSON Call ---")
    print(f"[AZURE] User message preview: {user_msg[:200]}...")
    try:
        response = client.chat.completions.create(
            model=deployment,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=max_completion_tokens,
        )
        content = response.choices[0].message.content
        print(f"[AZURE] Raw response: {content}")

        # Clean only leading/trailing whitespace — DO NOT strip quotes or newlines inside JSON
        content_clean = content.strip()

        # Strip markdown code fences if present (```json ... ```)
        if content_clean.startswith("```"):
            content_clean = re.sub(r"^```(?:json)?\s*", "", content_clean)
            content_clean = re.sub(r"\s*```$", "", content_clean)
            content_clean = content_clean.strip()

        parsed = json.loads(content_clean)

        # Handle double-encoded JSON (string containing JSON)
        if isinstance(parsed, str):
            print("[AZURE] Response was double-encoded JSON, parsing again...")
            parsed = json.loads(parsed)

        print(f"[AZURE] Parsed JSON: {parsed}")
        return parsed

    except json.JSONDecodeError as e:
        print(f"[AZURE] JSON decode error: {str(e)}")
        print(f"[AZURE] Content that failed to parse: {content!r}")
        return {}
    except Exception as e:
        print(f"[AZURE] ERROR during API call: {type(e).__name__}: {str(e)}")
        print(f"[AZURE] Full traceback:\n{traceback.format_exc()}")
        return {}

def classify_and_extract(query):
    print(f"\n[ROUTER] Classifying query: {query!r}")
    parsed = azure_json_call(
        "You are a structured data extractor for academic advising. Return only valid JSON with no markdown.",
        ROUTER_AND_FILTER_PROMPT.format(query=query),
        max_completion_tokens=500
    )

    if not parsed:
        print("[ROUTER] Empty parsed result, defaulting to KB_ONLY")
        return {"intent": "KB_ONLY", "levels": None, "topic": None}

    intent = parsed.get("intent", "KB_ONLY")
    if intent not in ["COURSE_ONLY", "KB_ONLY", "HYBRID"]:
        print(f"[ROUTER] Invalid intent '{intent}', defaulting to KB_ONLY")
        intent = "KB_ONLY"

    levels = parsed.get("levels")
    topic = parsed.get("topic")

    print(f"[ROUTER] Intent: {intent} | Levels: {levels} | Topic: {topic}")
    return {"intent": intent, "levels": levels, "topic": topic}

def extract_course_number(course_num_str):
    match = re.search(r'\d+', str(course_num_str))
    return int(match.group()) if match else None

def filter_by_levels(courses, levels):
    if not levels:
        return courses
    print(f"[COURSES] Filtering {len(courses)} courses by levels: {levels}")
    filtered = []
    for course in courses:
        num = extract_course_number(course["courseNum"])
        if num is None:
            continue
        for level in levels:
            level_int = int(level)
            if level_int <= num < level_int + 100:
                filtered.append(course)
                break
    print(f"[COURSES] After level filter: {len(filtered)} courses")
    return filtered

def keyword_topic_filter(topic, courses):
    if not topic:
        return courses
    topic_lower = topic.lower()
    matched = []
    for c in courses:
        title = (c.get("courseTitle") or "").lower()
        if topic_lower in title:
            matched.append(c)
    print(f"[COURSES] Keyword filter for '{topic}': {len(matched)} matches")
    return matched

def semantic_topic_filter(topic, courses):
    if not topic or not courses:
        return courses
    print(f"[COURSES] Running semantic filter for topic: '{topic}' on {len(courses)} courses")
    slim_courses = [{"courseNum": c["courseNum"], "courseTitle": c["courseTitle"]} for c in courses]
    parsed = azure_json_call(
        "You are a course relevance classifier. Return only valid JSON with no markdown.",
        LLM_TOPIC_FILTER_PROMPT.format(
            topic=topic,
            courses_json=json.dumps(slim_courses)
        ),
        max_completion_tokens=3000
    )
    relevant_nums = parsed.get("relevant_course_nums")
    if not relevant_nums:
        print("[COURSES] Semantic filter returned empty, falling back to all courses")
        return courses
    relevant_set = set(relevant_nums)
    filtered = [c for c in courses if c["courseNum"] in relevant_set]
    print(f"[COURSES] Semantic filter result: {len(filtered)} relevant courses")
    return filtered

def load_filtered_courses(levels, topic):
    try:
        print(f"\n[COURSES] Loading courses — levels: {levels}, topic: {topic}")
        response = supabase.table("Courses").select(
            "courseNum, courseTitle, units, prerequisites"
        ).execute()
        courses = response.data or []
        print(f"[COURSES] Loaded {len(courses)} total courses from Supabase")
        if courses:
            print(f"[COURSES] Sample: {courses[:2]}")
    except Exception as e:
        print(f"[COURSES] ERROR loading courses: {str(e)}")
        return []

    courses = filter_by_levels(courses, levels)

    if not topic:
        return courses

    keyword_matches = keyword_topic_filter(topic, courses)
    if keyword_matches:
        return keyword_matches

    return semantic_topic_filter(topic, courses)

def answer_course_query(levels, topic, user_context=""):
    courses = load_filtered_courses(levels, topic)
    if not courses:
        return "No matching courses found for your query."
    lines = []
    for c in courses:
        line = f"**{c['courseNum']}** - {c['courseTitle']} ({c['units']} units)"
        if c.get("prerequisites"):
            line += f"\n  *Prerequisites: {c['prerequisites']}*"
        lines.append(line)
    note = f"\n\n_{user_context}_" if user_context else ""
    return f"Here are matching courses ({len(courses)} found):\n\n" + "\n\n".join(lines) + note

def answer_kb_query(query, user_context=''):
    print(f"\n[KB ANSWER] Answering KB query: {query!r}")
    if not KB_CACHE:
        print("[KB ANSWER] KB cache is empty!")
        return "No knowledge base entries found. Please contact bellardo@calpoly.edu for more information."

    # Format KB entries using correct schema: title + content
    kb_entries = []
    for entry in KB_CACHE[:50]:
        title = entry.get("title", "Untitled")
        content = entry.get("content", "")
        source = entry.get("sourceURL", "")
        tags = entry.get("tags", [])
        kb_entries.append(f"### {title}\n{content}\nSource: {source}\nTags: {', '.join(tags) if tags else 'N/A'}")

    kb_context = "\n\n---\n\n".join(kb_entries)
    print(f"[KB ANSWER] Using {len(KB_CACHE[:50])} KB entries, context length: {len(kb_context)} chars")

    system_msg = "You are an academic advisor assistant for Cal Poly's graduate CS program."
    user_msg = KB_ANSWER_PROMPT.format(
        user_context=user_context,
        knowledge_context=kb_context,
        query=query
    )

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_completion_tokens=4000
        )
        content = response.choices[0].message.content
        print(f"[KB ANSWER] Response received ({len(content)} chars)")
        return content
    except Exception as e:
        print(f"[KB ANSWER] ERROR: {str(e)}")
        return f"Could not retrieve answer from knowledge base. Error: {str(e)}"



def run_degree_planning_agent(query, user_id):
    print(f"\n{'='*50}")
    print(f"[AGENT] New query: {query!r}")
    print(f"{'='*50}")

    parsed = classify_and_extract(query)
    intent = parsed["intent"]
    levels = parsed["levels"]
    topic = parsed["topic"]

    print(f"[AGENT] Routing to intent: {intent}")
    
    user = load_user_context(user_id) if user_id else None
    user_context = format_user_context(user)

    if intent == "COURSE_ONLY":
        result = answer_course_query(levels, topic, user_context)
    elif intent == "KB_ONLY":
        result = answer_kb_query(query, user_context)
    elif intent == "HYBRID":
        kb_answer = answer_kb_query(query, user_context)
        course_answer = answer_course_query(levels, topic, user_context)
        result = f"{kb_answer}\n\n---\n\n{course_answer}"
    else:
        result = "Could not determine intent. Please try rephrasing your question."

    print(f"[AGENT] Response length: {len(result)} chars")
    return result


def chat_with_gpt(message, history):
    if history is None:
        history = []
    history.append(gr.ChatMessage(role="user", content=message))
    try:
        response = run_degree_planning_agent(message)
    except Exception as e:
        response = f"Error: {str(e)}"
        print(f"[UI] Top-level error: {str(e)}")
    history.append(gr.ChatMessage(role="assistant", content=response))
    return history, ""


with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Grad-GPT")
    chatbot = gr.Chatbot(height=400)
    chat_input = gr.Textbox(
        placeholder="Ask about courses, requirements, faculty, and more...")
    send_button = gr.Button("Send")
    send_button.click(chat_with_gpt, [chat_input, chatbot], [
                      chatbot, chat_input])
    chat_input.submit(chat_with_gpt, [chat_input, chatbot], [
                      chatbot, chat_input])

demo.launch()
