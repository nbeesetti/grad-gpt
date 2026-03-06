# coordinator.py

import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from degree_agent import run_degree_planning_agent
from deadlines_agent import run_forms_and_deadlines_agent
from resource_agent import ResourceAgent

# Load environment variables
load_dotenv()

res_agent = ResourceAgent()

# Azure OpenAI Configuration
endpoint = "https://gradgpt-openai.openai.azure.com/"
deployment = "gradgpt-chat"
api_version = "2024-12-01-preview"
subscription_key = os.environ.get("Azure_API_Key")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# =========================
# SYSTEM PROMPTS
# =========================

SYSTEM_MESSAGE = """
You are the coordinator agent Grad-GPT for graduate students at Cal Poly SLO.
Some background information -- You can assume that anyone you
interact with is either a) a current CS MS student or b) a prospective CS MS student at Cal Poly SLO. Further, for current students, the Computer Science MS requires a thesis so don't ask if they are doing a thesis or not.  

You work alongside three specialized agents:
1. forms_agent → handles administrative forms, deadlines, graduation term requirements.
2. degree_planning_agent → handles course planning and degree progress.
3. resource_agent → handles research, literature, and compute resources.

You must follow this reasoning process internally:

Step 1: Determine user intent category.
Step 2: Determine if clarification is needed.
Step 3: Determine which agent(s) to call.
Step 4: If multi-agent, decompose the question into subqueries.
Step 5: If ready, delegate.

Rules:

- If clarification is needed, respond conversationally and ask a clarifying question.
- If the question is clear and ready for delegation, DO NOT respond conversationally.
- Instead, output ONLY valid JSON in the following format:
- Keep the query concise and only include information that was requested by the student, do not ask for more information than what the student asked for in the original query. If you need to ask for more information, ask a clarifying question instead of delegating.

{
  "delegate": true,
  "subqueries": [
    {
      "agent": "forms_agent | degree_planning_agent | resource_agent",
      "query": "sub-question text"
    }
  ]
}

- If multiple agents are needed, include multiple subqueries.
- Do not include any extra commentary when outputting JSON.
- If you are still clarifying, respond normally in natural language.
"""


SYNTHESIS_SYSTEM_MESSAGE = """
You are Grad-GPT, the coordinator agent.

You have received responses from specialized agents.
Your job is to combine their information into a single, clear,
well-structured, student-friendly response.

Some background information -- You can assume that anyone you
interact with is either a) a current CS MS student or b) a prospective CS MS student at Cal Poly SLO. Further, for current students, the Computer Science MS requires a thesis so don't ask if they are doing a thesis or not.

Rules:
- Do not mention agents.
- Do not mention delegation.
- Integrate the information logically.
- Use clean formatting.
- Be concise but helpful.
- If multiple topics were addressed, organize with headings.
- Do not include any information that was not provided by the agents.
- Do not ask for any additional information from the student. Only include information that was provided by the agents in your response.
"""


def ask_coordinator(user_message, history):

    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]

    # history is already in correct dict format
    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        messages=messages,
        max_completion_tokens=1024,
        model=deployment
    )

    return response.choices[0].message.content


def synthesize_response(original_user_query, agent_responses):

    formatted_input = f"""
    The student asked:
    {original_user_query}

    The following information was gathered from specialized systems:

    ---------------------
    {agent_responses}
    ---------------------

    Combine this information into a clear, organized response.
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_MESSAGE},
            {"role": "user", "content": formatted_input},
        ],
        max_completion_tokens=1500,
        model=deployment
    )

    print("Raw synthesis response:", response)

    return response.choices[0].message.content


# DELEGATION HANDLER


def handle_delegation(parsed_json, original_query, user_id=None):
    responses = []

    for sub in parsed_json["subqueries"]:
        agent = sub["agent"]
        query = sub["query"]

        if agent == "forms_agent":
            response = run_forms_and_deadlines_agent(query)

        elif agent == "degree_planning_agent":
            response = run_degree_planning_agent(query, user_id=user_id)  # pass user_id here

        elif agent == "resource_agent":
            #
            structured = res_agent.run_structured(query)
            ranked_resources = structured.get("ranked", [])
            if ranked_resources:
                lines = []
                for i, r in enumerate(ranked_resources, start=1):
                    lines.append(
                        f"{i}. {r.get('title', 'Untitled')}\n"
                        f"   Link: {r.get('link', '')}\n"
                        f"   Source: {r.get('source', '')}\n"
                        f"   Why: {r.get('why', '')}"
                    )
                response = "\n".join(lines)
            else:
                response = structured.get(
                    "message", "No relevant resources found.")

        else:
            response = "Unknown agent."

        responses.append(response)

    combined = "\n\n".join(responses)

    return synthesize_response(original_query, combined)


# MAIN entry


def process_message(user_message, history, user_id=None):
    """
    Main function to be called by Gradio UI.
    Returns updated history list.
    """

    if history is None:
        history = []

    assistant_response = ask_coordinator(user_message, history)

    # Append user message first
    history.append({"role": "user", "content": user_message})

    # Debug: show exactly what the coordinator returned
    print("[DEBUG] Coordinator output:", assistant_response)

    # Try parsing JSON to see if delegation is required
    try:
        parsed = json.loads(assistant_response)
        print("[DEBUG] Parsed JSON from coordinator:", parsed)

        if parsed.get("delegate") is True:
            final_response = handle_delegation(parsed, user_message, user_id=user_id)  # pass user_id here
            history.append({"role": "assistant", "content": final_response})
            return history

    except json.JSONDecodeError:
        # Not JSON → coordinator is clarifying or conversational
        print("[DEBUG] Coordinator response is conversational, not delegation JSON.")

    # Normal response (clarification)
    history.append({"role": "assistant", "content": assistant_response})
    return history
