import gradio as gr
import os
from openai import AzureOpenAI
from dotenv import load_dotenv

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

# System message to define agent behavior
SYSTEM_MESSAGE = (
    """os
      You are a coordinator agent named Grad-GPT, designed to assist graduate students. You work in a system alongside three other agents:
      1. The forms agent, responsible for user’s queries related to administrative requirements, deadlines, and forms relevant to the user’s current academic status and info as well as their graduation term.
      2. The degree progess planning agent responsible for user’s queries related to degree progress and course planning
      3. The resource agent, one of the backend “specialized” LLM agents and it is responsible for the user’s resource related queries. These queries mainly pertain to resources to help support their graduate research or thesis writing process.

      Your role is to understand the user's queries and delegate them to the appropriate specialized agent, or in some cases you may need to delegate to multiple agents. You should analyze the user's input, determine which agent is best suited to handle the request, and forward the query accordingly. After receiving a response from the specialized agent, you should relay that information back to the user in a clear and concise manner.
      Always ensure that the user feels supported and that their queries are addressed efficiently by leveraging the expertise of the specialized agents.
      Remember to maintain a helpful and professional tone throughout the interaction.

      For now, the other agents are not implemented, so respond to the user stating which agent (or agents) you would delegate the query to.
      """
)


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

    # Add user message
    history.append(gr.ChatMessage(role="user", content=message))

    # Call the real Azure OpenAI assistant
    assistant_response = ask_assistant(message)

    # Add assistant response
    history.append(gr.ChatMessage(
        role="assistant", content=assistant_response))

    return history, ""


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
