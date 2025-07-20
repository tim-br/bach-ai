# Bach-AI: Soprano and Bass Chorale Generator

A generative AI project that composes two-part harmony in the style of J.S. Bach. Given a soprano melody, the system learns to generate a counterpoint bass line using LSTM neural networks trained on the JSB Chorales corpus.

---

## 📁 Project Structure

| File                                 | Description                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------- |
| `train.py`                           | Trains an LSTM model on soprano-only melodies                                   |
| `generate.py`                        | Generates soprano-only output from `chorale_lstm.pt`                            |
| `train_bass.py`                      | Trains a bass model conditioned on existing soprano lines                       |
| `generate_bass.py`                   | Generates a bass line given a soprano sequence and trained `bass_lstm.pt`       |
| `train_joint_bass_and_soprano.py`    | Trains a joint model that learns soprano + bass interaction in interleaved form |
| `generate_joint_bass_and_soprano.py` | Generates both soprano and bass lines using the joint model `joint_lstm.pt`     |
| `bach_dataset.py`                    | Loads JSB chorale data and processes them into token sequences                  |
| `model.py`                           | Contains the PyTorch `ChoraleLSTM` class definition                             |

---

## 🧪 Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
uv venv && source .venv/bin/activate
uv pip install -r uv.lock
```

To install dependencies manually:

```bash
uv pip install torch music21
```

---

## 🚦 Training

### Soprano-only:

```bash
python train.py
```

Saves model to `chorale_lstm.pt`

### Bass conditioned on soprano:

```bash
python train_bass.py
```

Saves model to `bass_lstm.pt`

### Joint Soprano + Bass modeling:

```bash
python train_joint_bass_and_soprano.py
```

Saves model to `joint_lstm.pt`

---

## 🎼 Generation

### Generate Soprano Only:

```bash
python generate.py
```

Creates `output/generated.mid` and `generated.xml`

### Generate Bass from Soprano:

```bash
python generate_bass.py
```

Loads `chorale_lstm_with_keys.pt` and `bass_lstm.pt`, creates `output/soprano_bass.mid`

### Generate Joint Soprano + Bass:

```bash
python generate_joint_bass_and_soprano.py
```

Parses `generated.xml`, completes with bass, saves to `output/joint_generated.mid/xml`

---

## 🎧 Examples

Inside `examples/` you'll find chronological generations:

| File                                             | Description                                                                        |
| ------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `generated.xml/mid`                              | Soprano-only melody                                                                |
| `soprano_bass_old.mid/xml` → `_old_1` → `_old_4` | Early attempts at melody-conditioned bass generation                               |
| `joint_generated.xml/mid`                        | ✅ Best result to date — generated using the interleaved Soprano + Bass joint model |

---

## 📚 Dependencies

* PyTorch
* music21
* uv (package manager)

---

## 🚀 TODO

* Add Alto and Tenor for full SATB
* Visualization (piano roll / staff preview)
* Longer-form chorale generation
* Music-theory-aware loss constraints

---

## 🧠 Credits

Built by [Timothy Williams](https://chat.openai.com/) using open-source data from [music21](https://web.mit.edu/music21/).

Inspired by J.S. Bach's chorale harmonizations.

---
