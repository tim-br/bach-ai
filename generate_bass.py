from music21 import converter
import os
import torch
from music21 import stream, note, metadata
from model import ChoraleLSTM
from bach_dataset import load_soprano_bass_pairs, build_vocab

SEQ_LEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_soprano_from_xml(xml_path, quarterLength=0.25):
    score = converter.parse(xml_path)
    soprano = score.parts[0]  # assuming first part is soprano
    soprano_notes = soprano.flatten().notesAndRests

    key = score.analyze('key')
    key_token = f"KEY_{key.tonic.name.replace('-', 'b')}_{'MINOR' if key.mode == 'minor' else 'MAJOR'}"

    sequence = [key_token]
    for n in soprano_notes:
        pitch = n.pitch.midi if isinstance(n, note.Note) else -1
        steps = int(n.quarterLength / quarterLength)
        sequence.extend([pitch] * steps)

    return sequence


def generate_bass(model, soprano_seq, tok2idx, idx2tok):
    model.eval()
    tokens = [tok2idx[p] for p in soprano_seq if p in tok2idx]
    result = []

    for i in range(SEQ_LEN, len(tokens)):
        input_seq = tokens[i - SEQ_LEN:i]
        inp = torch.tensor([input_seq]).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
        probs = torch.softmax(logits[0], dim=0)
        next_tok = torch.multinomial(probs, 1).item()
        result.append(next_tok)

    return [idx2tok[i] for i in result]


def two_voice_stream(soprano_seq, bass_seq, quarterLength=0.25):
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Soprano + Generated Bass"

    soprano_part = stream.Part()
    bass_part = stream.Part()

    for part_seq, part in [(soprano_seq, soprano_part), (bass_seq, bass_part)]:
        for tok in part_seq:
            if isinstance(tok, str) and tok.startswith("KEY_"):
                continue
            if tok == -1:
                n = note.Rest(quarterLength=quarterLength)
            else:
                n = note.Note(midi=tok, quarterLength=quarterLength)
            part.append(n)

    score.append(soprano_part)
    score.append(bass_part)
    return score


def main():
    # Load vocab
    pairs = load_soprano_bass_pairs()
    tok2idx, idx2tok = build_vocab(
        [s for s, b in pairs] + [b for s, b in pairs])

    # Load model
    model = ChoraleLSTM(len(tok2idx))
    model.load_state_dict(torch.load("bass_lstm.pt", map_location=DEVICE))
    model.to(DEVICE)

    # Provide soprano sequence (example or load from previous generation)
    soprano_seq = extract_soprano_from_xml("output/generated.xml")

    # Generate bass line
    bass_seq = generate_bass(model, soprano_seq, tok2idx, idx2tok)

    # Combine and save
    score = two_voice_stream(soprano_seq, bass_seq)
    os.makedirs("output", exist_ok=True)
    score.write("midi", fp="output/soprano_bass.mid")
    score.write("musicxml", fp="output/soprano_bass.xml")
    print("Saved soprano_bass.mid and soprano_bass.xml")


if __name__ == "__main__":
    main()
