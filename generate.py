# generate.py

import torch
from model import ChoraleLSTM
from bach_dataset import build_vocab, load_chorales_soprano
from music21 import stream, note
import os

SEQ_LEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate(model, seed, length, idx2tok):
    model.eval()
    result = seed[:]
    for _ in range(length):
        inp = torch.tensor([result[-SEQ_LEN:]], device=DEVICE)
        with torch.no_grad():
            logits = model(inp)
        probs = torch.softmax(logits[0], dim=0)
        next_token = torch.multinomial(probs, 1).item()
        result.append(next_token)
    return [idx2tok[i] for i in result]


def sequence_to_stream(seq, quarterLength=0.25):
    s = stream.Stream()
    for p in seq:
        if isinstance(p, str) and p.startswith("KEY_"):
            continue  # Skip key tokens
        if p == -1:
            n = note.Rest(quarterLength=quarterLength)
        else:
            n = note.Note(midi=p, quarterLength=quarterLength)
        s.append(n)
    return s


def main():
    sequences = load_chorales_soprano()
    tok2idx, idx2tok = build_vocab(sequences)

    model = ChoraleLSTM(len(tok2idx))
    model.load_state_dict(torch.load(
        "chorale_lstm_with_keys.pt", map_location=DEVICE))

    model.to(DEVICE)

    # Choose a key token that exists in your vocab
    key_token = "KEY_C_MAJOR"
    if key_token not in tok2idx:
        raise ValueError(f"{key_token} not in vocab!")

    seed_seq = [tok2idx[key_token]]

    # Optionally follow with real pitches (e.g. from training data)
    # skip the key token in the source
    seed_seq += [tok2idx[p] for p in sequences[0][1:SEQ_LEN]]

    output = generate(model, seed_seq, 100, idx2tok)

    s = sequence_to_stream(output)
    os.makedirs("output", exist_ok=True)
    s.write('midi', fp='output/generated.mid')
    s.write('musicxml', fp='output/generated.xml')
    s.show('midi')  # or s.show() to open notation view


if __name__ == "__main__":
    main()
