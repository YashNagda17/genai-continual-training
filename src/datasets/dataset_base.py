b"""Implement source-specific processing here, independently of paper policies."""

from abc import ABC, abstractmethod


class DatasetBase(ABC):
    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """Preprocessing settings as plain JSON."""

    @abstractmethod
    def load(self, source_spec):
        """Load a source or saved manifest; verify provenance and referenced assets."""

    @abstractmethod
    def split(self, raw_view, split_spec):
        """Return immutable, disjoint membership; preserve official splits."""

    @abstractmethod
    def preprocess(self, view, preprocessing_spec):
        """Apply source formatting/tokenization while preserving observation IDs and targets.
        Report dropped or truncated records so membership and token accounting remain reviewable.
        """

    @abstractmethod
    def training_view(self, task_id):
        """Return the requested task's immutable finite training records."""

    @abstractmethod
    def evaluation_view(self, task_id):
        """Return held-out records, accessible only through evaluation."""

    @abstractmethod
    def collate(self, records):
        """Build a Batch with Torch inputs, targets and supervision masks, normally on CPU.
        Preserve record order and IDs, including repeated IDs from replay sampling.
        Keep valid/supervised-token counts explicit; the model owns device and precision changes.
        """

    @abstractmethod
    def save(self, path):
        """Persist data manifests and permitted artifacts, without replay policy."""
