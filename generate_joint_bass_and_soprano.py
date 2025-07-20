import os
import torch
from music21 import converter, stream, note, metadata
from model import ChoraleLSTM
from bach_dataset import build_vocab, load_soprano_bass_pairs

SEQ_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REST = -1


def extract_soprano_from_xml(xml_path, quarterLength=0.25):
    score = converter.parse(xml_path)
    soprano = score.parts[0]  # assuming soprano is first part
    soprano_notes = soprano.flatten().notesAndRests
    key = score.analyze('key')
    key_token = f"KEY_{key.tonic.name.replace('-', 'b')}_{'MINOR' if key.mode == 'minor' else 'MAJOR'}"
    sequence = [key_token]
    for n in soprano_notes:
        pitch = n.pitch.midi if isinstance(n, note.Note) else REST
        steps = int(n.quarterLength / quarterLength)
        sequence.extend([pitch] * steps)
    return sequence


def interleave_with_rest(soprano_seq):
    result = [soprano_seq[0]]  # key
    for s in soprano_seq[1:]:
        result.append(s)
        result.append(REST)  # unknown bass
    return result


def split_interleaved(seq):
    soprano, bass = [], []
    for i, tok in enumerate(seq):
        if isinstance(tok, str) and tok.startswith("KEY_"):
            continue
        (soprano if i % 2 == 0 else bass).append(tok)
    return soprano, bass


def sequence_to_stream(soprano_seq, bass_seq, quarterLength=0.25):
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Generated Soprano and Bass"

    soprano_part = stream.Part()
    bass_part = stream.Part()

    for seq, part in [(soprano_seq, soprano_part), (bass_seq, bass_part)]:
        for tok in seq:
            if tok == REST:
                n = note.Rest(quarterLength=quarterLength)
            else:
                n = note.Note(midi=tok, quarterLength=quarterLength)
            part.append(n)

    score.append(soprano_part)
    score.append(bass_part)
    return score


def generate_joint():
    # Load vocab
    pairs = load_soprano_bass_pairs()
    tok2idx, idx2tok = build_vocab(
        [s for s, _ in pairs] + [b for _, b in pairs])

    # Load model
    model = ChoraleLSTM(len(tok2idx)).to(DEVICE)
    model.load_state_dict(torch.load("joint_lstm.pt", map_location=DEVICE))
    model.eval()

    # Load soprano from xml
    soprano_seq = extract_soprano_from_xml("output/generated.xml")
    interleaved = interleave_with_rest(soprano_seq)

    # Map to token indices
    tokens = [tok2idx[p] for p in interleaved if p in tok2idx]

    result = tokens[:SEQ_LEN]  # start with seed
    while len(result) < len(tokens):
        input_seq = result[-SEQ_LEN:]
        inp = torch.tensor([input_seq]).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
        next_tok = torch.multinomial(torch.softmax(logits[0], dim=0), 1).item()
        result.append(next_tok)

    # Convert back to tokens
    full_seq = [idx2tok[i] for i in result]
    soprano_gen, bass_gen = split_interleaved(full_seq)
    score = sequence_to_stream(soprano_gen, bass_gen)

    os.makedirs("output", exist_ok=True)
    score.write("midi", fp="output/joint_generated.mid")
    score.write("musicxml", fp="output/joint_generated.xml")
    print("Saved output/joint_generated.mid and .xml")


if __name__ == "__main__":
    generate_joint()
