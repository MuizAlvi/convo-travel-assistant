import gradio as gr
from main import travel_assistant_response

def chat_interface(user_input, history):
    response = travel_assistant_response(user_input)
    history.append((user_input, response))
    return "", history

with gr.Blocks(title="Travel Assistant") as app:
    gr.Markdown("## Conversational Travel Assistant")
    gr.Markdown("Ask me about flights, visas, or refund policies.")

    chatbot = gr.Chatbot()
    state = gr.State([])

    with gr.Row():
        user_input = gr.Textbox(placeholder="Type your travel question...")
        send_btn = gr.Button("Send")

    send_btn.click(chat_interface, inputs=[user_input, state], outputs=[user_input, chatbot])
    user_input.submit(chat_interface, inputs=[user_input, state], outputs=[user_input, chatbot])

if __name__ == "__main__":
    app.launch(server_port=7860)

### Note: Go to http://localhost:7860 to view the app.