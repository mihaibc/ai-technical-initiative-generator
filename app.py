import os

import gradio as gr
from dotenv import load_dotenv

from aitoolkit.ui import build_ui


def main():
    load_dotenv(override=False)
    demo = build_ui()
    server_port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=server_port, show_error=True)


if __name__ == "__main__":
    main()
