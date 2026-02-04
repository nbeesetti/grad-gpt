import gradio as gr


def chat_with_gpt(message, history):
    if history is None:
        history = []

    # add user message
    history.append(gr.ChatMessage(role="user", content=message))

    # fake assistant reply
    msg_lower = message.lower()
    if "course" in msg_lower or "courses" in msg_lower:
        assistant_response = f"Enroll in CSC 580 for amazing AI knowledge!"
    else:
        assistant_response = f"Grad Assistant says: {message}"
    history.append(gr.ChatMessage(
        role="assistant", content=assistant_response))

    return history, ""


with gr.Blocks() as demo:
    gr.Markdown("# Welcome to Grad-GPT")
    gr.Markdown("This is a demo application for Grad-GPT.")

    # Maybe other tabs can be added later? i.e. resources, course progress
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
