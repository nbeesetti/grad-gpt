import gradio as gr
import os
from openai import AzureOpenAI
from dotenv import load_dotenv
import json
from degree_agent import run_degree_planning_agent
# load env variables
load_dotenv()

# Azure OpenAI details
endpoint = "https://gradgpt-openai.openai.azure.com/"
deployment = "gradgpt-chat"
api_version = "2024-12-01-preview"
subscription_key = os.environ.get("Azure_API_Key")

# Instantiate client
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)


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


# System message for coordinator once receives responses from specialized agents
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


def ask_assistant(user_message: str):
    """
    Method to ask the Azure OpenAI assistant a question, return response
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=1024,
        model=deployment
    )
    return response.choices[0].message.content


def chat_with_gpt(message, history):
    if history is None:
        history = []

    history.append(gr.ChatMessage(role="user", content=message))

    assistant_response = ask_assistant(message)

    # Try parsing JSON delegation
    try:
        parsed = json.loads(assistant_response)

        if parsed.get("delegate") is True:
            # Call delegation handler
            final_response = handle_delegation(parsed, message)

            history.append(gr.ChatMessage(
                role="assistant", content=final_response))
            return history, ""

    except json.JSONDecodeError:
        pass  # Not JSON,  normal conversation/ keep asking clarifying questions until we get delegation JSON

    # Normal conversational response
    history.append(gr.ChatMessage(
        role="assistant", content=assistant_response))
    print(f"History: {history}")
    return history, ""


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

# TODO: Implement actual agent calls instead of placeholders


def forms_agent(query):
    return """
Graduation Application:
- Submit via the myportal
- Deadline: Week 2 of final quarter

Thesis Approval Form:
- Signed by committee
- Submit to Graduate Studies
"""


def degree_planning_agent(query):
    return """
Course Planning:
Take CSC 590, CSC 599
"""


def resource_agent(query):
    return """
Read the paper "Attention is All You Need" for transformer models.
Use Google Scholar for research.
Use arXiv for preprints.
"""


def handle_delegation(parsed_json, original_user_query):
    responses = []
    print(f"Parsed JSON: {parsed_json}")

    for sub in parsed_json["subqueries"]:
        agent = sub["agent"]
        query = sub["query"]

        if agent == "forms_agent":
            response = forms_agent(query)
        elif agent == "degree_planning_agent":
            response = run_degree_planning_agent(query)
        elif agent == "resource_agent":
            response = resource_agent(query)
        else:
            response = "Unknown agent."

        responses.append(response)

    combined_agent_output = "\n\n".join(responses)
    # TODO -- if confidence score low, could loop again or try to ask more questions to the user before synthesizing, but for now just go straight to synthesis

    print(f"Combined agent output before synthesis: {combined_agent_output}")
    # Now synthesize
    final_response = synthesize_response(
        original_user_query,
        combined_agent_output
    )
    print(f"Final synthesized response: {final_response}")

    return final_response


with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Grad-GPT")
    gr.Markdown("This is a demo application for Grad-GPT.")

    with gr.Tab("Chat with your Grad Assistant!"):
        chatbot = gr.Chatbot(label="Grad-GPT", height=400)

        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                scale=8
            )
            send_button = gr.Button("Send", scale=1)

        send_button.click(
            chat_with_gpt,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input],
        )

        chat_input.submit(
            chat_with_gpt,
            inputs=[chat_input, chatbot],
            outputs=[chatbot, chat_input],
        )

demo.launch()
