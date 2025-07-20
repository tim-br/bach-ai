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


def load_soprano_bass_pairs(quarterLength=0.25):
    pairs = []
    for chorale in corpus.chorales.Iterator():
        parts = chorale.parts.stream()
        if len(parts) != 4:
            continue

        soprano = parts[0].flatten().notesAndRests
        bass = parts[-1].flatten().notesAndRests

        key = chorale.analyze('key')
        key_token = f"KEY_{key.tonic.name.replace('-', 'b')}_{'MINOR' if key.mode == 'minor' else 'MAJOR'}"

        dur = chorale.highestTime
        steps = int(dur / quarterLength)
        sop_seq = [REST] * steps
        bass_seq = [REST] * steps

        for voice, stream_notes in enumerate([soprano, bass]):
            for n in stream_notes:
                t0 = int(n.offset / quarterLength)
                t_len = max(1, int(n.quarterLength / quarterLength))
                val = n.pitch.midi if isinstance(n, note.Note) else REST
                for i in range(t0, min(t0 + t_len, steps)):
                    if voice == 0:
                        sop_seq[i] = val
                    else:
                        bass_seq[i] = val

        # Optional: prepend key token to soprano input
        sop_seq = [key_token] + sop_seq
        bass_seq = [REST] + bass_seq  # shift target to match input length
        pairs.append((sop_seq, bass_seq))
    return pairs


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


class MelodyToBassDataset(Dataset):
    def __init__(self, pairs, tok2idx, seq_len):
        self.samples = []
        for soprano, bass in pairs:
            s_idx = [tok2idx[p] for p in soprano]
            b_idx = [tok2idx[p] for p in bass]
            for i in range(len(s_idx) - seq_len):
                x = s_idx[i:i+seq_len]
                y = b_idx[i+seq_len]  # predict next bass note
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)
