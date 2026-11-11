"""google/mt5-small: 8 + 8 blocks, 300,176,768 parameters."""

from .model_seq2seq import Seq2SeqLMModel


class MT5SmallModel(Seq2SeqLMModel):
    model_id = "google/mt5-small"
