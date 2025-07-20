import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bach_dataset import load_soprano_bass_pairs, build_vocab
from model import ChoraleLSTM

SEQ_LEN = 64
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class InterleavedSATBDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, tok2idx, seq_len):
        self.samples = []
        for soprano, bass in pairs:
            interleaved = []
            for s, b in zip(soprano[1:], bass[1:]):  # skip key/rest alignment
                interleaved.append(s)
                interleaved.append(b)
            interleaved = [soprano[0]] + interleaved  # prepend key
            idx_seq = [tok2idx[p] for p in interleaved if p in tok2idx]
            for i in range(len(idx_seq) - seq_len):
                x = idx_seq[i:i+seq_len]
                y = idx_seq[i+seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


def main():
    print("Loading soprano-bass pairs...")
    pairs = load_soprano_bass_pairs()
    print(f"Loaded {len(pairs)} chorales.")

    print("Building vocabulary...")
    tok2idx, idx2tok = build_vocab(
        [s for s, _ in pairs] + [b for _, b in pairs])

    print("Building interleaved dataset...")
    dataset = InterleavedSATBDataset(pairs, tok2idx, SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ChoraleLSTM(len(tok2idx)).to(DEVICE)

    if os.path.exists("joint_lstm.pt"):
        print("Loading existing weights...")
        model.load_state_dict(torch.load("joint_lstm.pt", map_location=DEVICE))

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    print("Training...")
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

    torch.save(model.state_dict(), "joint_lstm.pt")
    print("Saved model to joint_lstm.pt")


if __name__ == "__main__":
    main()
