import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bach_dataset import load_chorales_soprano, build_vocab, SopranoDataset
from model import ChoraleLSTM

SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Load data and vocab
    sequences = load_chorales_soprano()
    tok2idx, idx2tok = build_vocab(sequences)
    dataset = SopranoDataset(sequences, tok2idx, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = ChoraleLSTM(vocab_size=len(tok2idx)).to(DEVICE)

    # Load existing model weights if available
    if os.path.exists("chorale_lstm_with_keys.pt"):
        print("Loading existing model weights...")
        model.load_state_dict(torch.load(
            "chorale_lstm_with_keys.pt", map_location=DEVICE))

    # Initialize optimizer (not restored for simplicity)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss={total_loss / len(loader):.4f}")

    # Save updated model weights
    torch.save(model.state_dict(), "chorale_lstm_with_keys.pt")
    print("Model saved to chorale_lstm.pt")


if __name__ == "__main__":
    main()
