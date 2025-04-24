import os
import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from analyze_checkpoints import load_model, generate_text
from main import get_activation
import tiktoken
from full_analysis_pipeline import extract_config_from_dir


CHECKPOINT_DIR = "checkpoints"

def get_checkpoints():
    return [d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))]

def generate_from_ckpt(checkpoint_folder, epoch, prompt, max_new_tokens):
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    folder = os.path.join(CHECKPOINT_DIR, checkpoint_folder)

    # Load matching checkpoint path
    ckpt_file = f"epoch_{epoch}.pt"
    checkpoint_path = os.path.join(folder, ckpt_file)
    if not os.path.exists(checkpoint_path):
        return f"[❌] Checkpoint not found: {ckpt_file}", ""

    # Extract config from folder name
   # New logic
    config = extract_config_from_dir(checkpoint_folder)
    if config is None:
        return f"❌ Failed to parse config from: {checkpoint_folder}", ""

    model = load_model(
        model_type=config["model_type"],
        vocab_size=vocab_size,
        checkpoint_path=checkpoint_path,
        embed_size=config["embed_size"],
        k=config["k"],
        chunk_size=config["chunk_size"],
        num_inner_layers=config["inner_layers"],
        block_size=config["block_size"],
        activation=config["activation"]
    )

    gen_text, annotated = generate_text(
        model, enc, prompt, max_new_tokens=max_new_tokens,
        top_p=0.95, monosemantic_info=None, do_monosemantic=True
    )

    return gen_text, annotated


def load_plot_image(checkpoint_folder):
    path = os.path.join(CHECKPOINT_DIR, checkpoint_folder, "metrics_curve.png")
    return path if os.path.exists(path) else None


def launch_gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("# 🔍 Language Model Explorer (Gradio Edition)")

        with gr.Row():
            checkpoint_folder = gr.Dropdown(choices=get_checkpoints(), label="Checkpoint Folder")
            epoch = gr.Number(label="Epoch", value=0, precision=0)
            prompt = gr.Textbox(label="Prompt", value="Once upon a", lines=2)
            max_tokens = gr.Slider(label="Max Tokens", minimum=10, maximum=100, step=10, value=30)

        generate_btn = gr.Button("Generate")
        out_gen = gr.Textbox(label="Generated Text")
        out_anno = gr.Textbox(label="Annotated Output")

        with gr.Row():
            plot_img = gr.Image(label="📈 Training Curves")
            reload_btn = gr.Button("Refresh Plot")

        generate_btn.click(fn=generate_from_ckpt,
                           inputs=[checkpoint_folder, epoch, prompt, max_tokens],
                           outputs=[out_gen, out_anno])
        reload_btn.click(fn=load_plot_image, inputs=checkpoint_folder, outputs=plot_img)

    demo.launch()

if __name__ == "__main__":
    launch_gradio_app()
# The code above is a Gradio app that allows users to interactively explore language model checkpoints.