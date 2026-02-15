"""Gradio web UI for Wikipedia summarization."""

import gradio as gr
from config import DEFAULT_TARGET_TOKENS, MODEL_NAME, MIN_TARGET_TOKENS, MAX_TARGET_TOKENS, GRADIO_PORT
from summarizer import summarize_wiki
from model_loader import count_tokens


def launch_gradio():
    """Launch Gradio web interface (blocking)."""
    def gradio_wrapper(url: str, target_tokens: float) -> str:
        summary = summarize_wiki(url, int(target_tokens))
        tokens = count_tokens(summary)
        return f"{summary}\n\n[Token count: {tokens} tokens]"

    demo = gr.Interface(
        fn=gradio_wrapper,
        inputs=[
            gr.Textbox(label="Wikipedia URL", placeholder="https://en.wikipedia.org/wiki/..."),
            gr.Slider(
                minimum=MIN_TARGET_TOKENS, maximum=MAX_TARGET_TOKENS,
                value=DEFAULT_TARGET_TOKENS, step=10, label="Target Token Count",
            ),
        ],
        outputs=gr.Textbox(label="Summary"),
        title="Wikipedia Summarizer",
        description=f"Summarize Wikipedia articles using {MODEL_NAME}",
    )
    demo.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=False)
