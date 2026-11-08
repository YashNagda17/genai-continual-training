"""google/mt5-small. Transformers supplies the encoder-decoder; the executor supplies LoRA."""

from .model_seq2seq import Seq2SeqLMModel


class MT5SmallModel(Seq2SeqLMModel):
    model_id = "google/mt5-small"
    model_type = "mt5"
    untie_head = True
    expected_parameters = 300_176_768
