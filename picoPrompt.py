import torch
import tiktoken
from picomodels import LSTMSeqModel
from utils import PicoGenerate

def main():
    # Settings (must match training)
    model_name = "lstm_seq"
    ckpt_path = f"{model_name}.pt"
    embed_size = 512  # match training args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    # Load model
    model = LSTMSeqModel.LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Model '{model_name}' loaded and ready.")

    print("\nEnter a prompt (or 'exit' to quit):")
    while True:
        prompt = input(">>> ")
        if prompt.strip().lower() == "exit":
            break
        with torch.no_grad():
            text, annotated = PicoGenerate.generate_text(
                model=model,
                enc=enc,
                init_text=prompt,
                max_new_tokens=50,
                device=device,
                top_p=0.95,
            )
        print(f"\nGenerated:\n{text}\n")
        # print(f"Annotated:\n{annotated}\n")

if __name__ == "__main__":
    main()
