import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bach_dataset import load_soprano_bass_pairs, build_vocab, MelodyToBassDataset
from model import ChoraleLSTM

SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Load melody-bass training pairs
    print("Loading soprano-bass pairs...")
    pairs = load_soprano_bass_pairs()
    print(f"Loaded {len(pairs)} chorales.")

    # Build vocab from all tokens (soprano + bass)
    print("Building vocabulary...")
    tok2idx, idx2tok = build_vocab(
        [s for s, _ in pairs] + [b for _, b in pairs])

    # Build dataset and dataloader
    print("Building dataset...")
    dataset = MelodyToBassDataset(pairs, tok2idx, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = ChoraleLSTM(len(tok2idx)).to(DEVICE)

    # Load existing model weights (optional)
    if os.path.exists("bass_lstm.pt"):
        print("Loading existing model weights...")
        model.load_state_dict(torch.load("bass_lstm.pt", map_location=DEVICE))

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Starting training...")
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

    torch.save(model.state_dict(), "bass_lstm.pt")
    print("Saved model to bass_lstm.pt")


if __name__ == "__main__":
    main()
