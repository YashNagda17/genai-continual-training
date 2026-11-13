from pathlib import Path
import json
import random
import tempfile

from transformers import MBartConfig, MBartForConditionalGeneration

from ..datasets.dataset_instruction import Seq2SeqPreprocessing, SummarizationDataset
from ..datasets.dataset_sources import load_source
from ..models.model_baselines import PaperMBart
from ..models.model_seq2seq import Seq2SeqSettings


class SmokeTokenizer:
    """Pad is not the decoder-start id, so a tiny model that emits zeros still yields words."""

    pad_token_id = 2
    eos_token_id = 1
    unk_token_id = 3
    words = ("the", "ruling", "party", "won", "state", "vote", "today", "capital")

    def convert_tokens_to_ids(self, token):
        return 0 if token == 'en_XX' else self.unk_token_id

    def __len__(self):
        return 32

    def get_vocab(self):
        return {str(index): index for index in range(32)}

    def encode(self, text, add_special_tokens=True):
        ids = [4 + (sum(ord(character) for character in word) % 27) for word in str(text).split()] or [4]
        return ids + ([self.eos_token_id] if add_special_tokens else [])

    def decode(self, ids, skip_special_tokens=True):
        skip = {self.pad_token_id, self.eos_token_id, self.unk_token_id} if skip_special_tokens else set()
        return " ".join(self.words[int(index) % len(self.words)] for index in ids if int(index) not in skip)

    def batch_decode(self, rows, skip_special_tokens=True):
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in rows]


class SmokeModel(PaperMBart):
    model_id = "summarization-smoke"


def _seeded_smoke_module(config):
    """The tiny model's random weights come from a fixed seed, not the process-wide generator,
    so a smoke run starts from the same weights whatever ran before it."""
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return MBartForConditionalGeneration(config)


def build_smoke_model(device="cpu"):
    settings = Seq2SeqSettings(rank=4, alpha=8.0, learning_rate=0.05, max_source_length=16,
                               max_target_length=8, max_batch_size=4, seed=3)
    config = MBartConfig(vocab_size=32, d_model=32, encoder_ffn_dim=64, decoder_ffn_dim=64,
                         encoder_attention_heads=4, decoder_attention_heads=4, encoder_layers=2,
                         decoder_layers=2, dropout=0., attention_dropout=0., activation_dropout=0.,
                         decoder_start_token_id=0, eos_token_id=1, pad_token_id=2)
    role = SmokeModel(settings, device=device, pretrained=False, config=config,
                      tokenizer=SmokeTokenizer(), model_factory=_seeded_smoke_module)
    role.runtime_root = Path(tempfile.mkdtemp())
    return role


def build_smoke_dataset(model):
    """A few English and Hindi rows, prepared with the smoke tokenizer. No download."""
    articles = {"english": "The ruling party won the state vote today in the capital",
                "hindi": "भारत में बड़ी खबर आई आज सुबह शहर में"}
    summaries = {"english": "The ruling party won the state vote", "hindi": "भारत में बड़ी खबर आई आज"}
    rows = []
    for split, count in (("train", 4), ("validation", 2), ("test", 2)):
        for language in ("english", "hindi"):
            for index in range(count):
                rows.append({"id": f"{language}/{split}-{index}", "task": language, "split": split,
                             "language": language, "text": f"{articles[language]} {index}",
                             "summary": summaries[language]})
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        with (folder / "records.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (folder / "manifest.json").write_text(json.dumps(
            {"dataset_id": "smoke", "revision": "smoke", "schema": "summarization"}), encoding="utf-8")
        dataset, raw = load_source(folder, SummarizationDataset)
    dataset.split(raw, {})
    dataset.preprocess(dataset.views, Seq2SeqPreprocessing(**model.preprocessing_kwargs(), prefix=None))
    dataset.expose("english/pool", [record.sample_id for record in dataset.training_view("english")])
    return dataset


def with_prefix(dataset, model, prefix):
    """Prefix exactly the registered subsets, preserving all ordered protocol views."""
    fresh = SummarizationDataset()
    fresh.manifest = dict(dataset.manifest)
    views = {split: tuple(row for language in ("english", "hindi") for row in
                 dataset.views.get(f"{language}/{split}", tuple(r for r in dataset.views.get(split, ()) if r.task_id == language)))
             for split in ("train", "validation", "test")}
    fresh.preprocess(views, Seq2SeqPreprocessing(**model.preprocessing_kwargs(), prefix=prefix))
    for name, ids in dataset.manifest.get("protocol", {}).items():
        fresh.expose(name, ids)
    return fresh


def choose_ids(records, count, seed):
    """Draw a stable subset and return it in arrival order."""
    ordered = sorted(records, key=lambda record: (record.arrival, record.sample_id))
    picked = set(random.Random(seed).sample([record.sample_id for record in ordered], count))
    return [record.sample_id for record in ordered if record.sample_id in picked]


def xlsum_directory():
    root = Path(__file__).resolve().parents[1] / "datasets" / "data" / "xlsum"
    found = [path for path in root.iterdir() if (path / "records.jsonl").is_file()] if root.is_dir() else []
    if len(found) != 1:
        raise SystemExit("XL-Sum is not fetched. Run ./download_datasets.sh xlsum")
    return found[0]
