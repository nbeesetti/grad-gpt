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

Background:
Assume every user is either:
a) a current CS MS student, or
b) a prospective CS MS student at Cal Poly SLO.

For current students, the CS MS program requires a thesis, so never ask whether they are doing a thesis.

You work alongside three specialized agents:

1. forms_agent → handles administrative forms, deadlines, graduation requirements, and official procedures.
2. degree_planning_agent → handles course planning, program requirements, and degree progress.
3. resource_agent → handles research resources, literature searches, datasets, compute tools, and project tools.

Your primary job is to ROUTE the user's request to the correct agent.

IMPORTANT PRINCIPLE:
When a user asks a question, you should almost always delegate the request to an agent immediately.

Do NOT ask follow-up questions unless delegation is impossible.

Uncertainty about details should be passed to the downstream agent instead of asking the user.

Only ask a clarification question if BOTH of the following are true:
1. The question cannot be understood.
2. The request cannot reasonably be routed to any agent.

Otherwise, delegate.

Reasoning process (internal only):
1. Identify the user intent.
2. Choose the most appropriate agent.
3. If the question contains multiple intents, split it into subqueries.
4. Delegate.

Delegation Rules:

If the question is understandable and can be routed, output ONLY valid JSON in this format:

{
  "delegate": true,
  "subqueries": [
    {
      "agent": "forms_agent | degree_planning_agent | resource_agent",
      "query": "concise reformulation of the user's question"
    }
  ]
}

Rules for writing subqueries:
- Keep the query concise.
- Preserve the student's original intent.
- Do not add new questions.
- Do not request extra information.
- Do not ask the user for clarification.
- If information is missing, pass the question as-is to the agent.

Multiple agents:
If the question involves multiple domains, create multiple subqueries.

Examples:

User: "What classes should I take next quarter?"
→ degree_planning_agent

User: "What forms to fill out before graduation?"
→ forms_agent

User: "What tools can I use to search research papers?"
→ resource_agent

User: "What classes should I take and when is the graduation application due?"
→ degree_planning_agent + forms_agent

If clarification is absolutely required, respond normally in natural language and ask concise clarification questions.
Otherwise, output ONLY JSON.
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
        max_completion_tokens=8000,
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
        model=deployment,
        max_completion_tokens=8000
    )

    print("Raw synthesis response:", response)

    return response.choices[0].message.content


# DELEGATION HANDLER


def handle_delegation(parsed_json, original_query, user_id=None):
    responses = []
    print(
        f"Handle Delegation with parsed json: {parsed_json} and OG query: {original_query}")
    for sub in parsed_json["subqueries"]:
        agent = sub["agent"]
        query = sub["query"]

        if agent == "forms_agent":
            response = run_forms_and_deadlines_agent(query)

        elif agent == "degree_planning_agent":
            response = run_degree_planning_agent(
                query, user_id=user_id)  # pass user_id here

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
    print(f"Combined Responses: {responses}")

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

    # Try parsing JSON to see if delegation is required
    delegated_agents = []

    try:
        parsed = json.loads(assistant_response)

        if parsed.get("delegate") is True:
            delegated_agents = [sub["agent"]
                                for sub in parsed.get("subqueries", [])]

            final_response = handle_delegation(
                parsed, user_message, user_id=user_id
            )

            history.append({"role": "assistant", "content": final_response})

        else:
            history.append(
                {"role": "assistant", "content": assistant_response})

    except json.JSONDecodeError:
        history.append({"role": "assistant", "content": assistant_response})

    # Output for coordinator routing testing

    print("\n" * 10)
    print("==============================================================")
    print("============== GRAD-GPT COORDINATOR ROUTING ====================")
    print("==============================================================")
    print("USER MESSAGE:")
    print(user_message)
    print("--------------------------------------------------------------")

    if delegated_agents:
        print("AGENTS DELEGATED TO:")
        for agent in delegated_agents:
            print(f"  - {agent}")
    else:
        print("AGENTS DELEGATED TO: NONE (coordinator handled or clarifying)")

    print("==============================================================")
    print("\n" * 5)

    return history
