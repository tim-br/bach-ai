from music21 import corpus, note, stream
import torch
from torch.utils.data import Dataset

REST = -1


def load_chorales_soprano(quarterLength=0.25):
    sequences = []
    for chorale in corpus.chorales.Iterator():
        parts = chorale.parts.stream()
        if len(parts) < 1:
            continue
        soprano = parts[0].flatten().notesAndRests

        # Extract key using music21
        key = chorale.analyze('key')
        key_token = f"KEY_{key.tonic.name.replace('-', 'b')}_{'MINOR' if key.mode == 'minor' else 'MAJOR'}"

        seq = [key_token]
        for e in soprano:
            if isinstance(e, note.Note):
                pitch = e.pitch.midi
            else:
                pitch = REST
            n_steps = int(e.quarterLength / quarterLength)
            seq.extend([pitch] * n_steps)
        sequences.append(seq)
    return sequences


def build_vocab(sequences):
    vocab = sorted({p for seq in sequences for p in seq}, key=str)
    tok2idx = {tok: i for i, tok in enumerate(vocab)}
    idx2tok = {i: tok for tok, i in tok2idx.items()}
    return tok2idx, idx2tok


class SopranoDataset(Dataset):
    def __init__(self, sequences, tok2idx, seq_len):
        self.samples = []
        for seq in sequences:
            idx_seq = [tok2idx[p] for p in seq]
            for i in range(len(idx_seq) - seq_len):
                x = idx_seq[i:i+seq_len]
                y = idx_seq[i+seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.tensor(x), torch.tensor(y)
