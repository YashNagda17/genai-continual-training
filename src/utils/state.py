"""Replay membership, weight snapshots, and checksummed checkpoint storage."""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shutil
import uuid

from .contracts import Record, training_records


def write_json(path, value):
    """Write and flush a temporary JSON file before atomically replacing the destination.
    Flush the parent directory after replacement so the new filename is durable.
    Readers therefore see a complete old or new file, rather than a partial write.
    """
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_json(path):
    """Read one persisted JSON value; callers decide its schema and compatibility."""
    with Path(path).open() as stream:
        return json.load(stream)


def checksums(path):
    """Hash each payload file relative to its artifact directory.
    Exclude the completion marker, which stores these hashes, and reject symbolic links.
    """
    result = {}
    for file in sorted(Path(path).rglob("*")):
        if file.is_symlink():
            raise ValueError("Checkpoint files must not be symlinks")
        if file.is_file() and file != Path(path) / "complete.json":
            with file.open("rb") as stream:
                result[str(file.relative_to(path))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def synchronize(path):
    """Flush model files and their directory before writing the completion marker."""
    for file in Path(path).rglob("*"):
        if file.is_file():
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Checkpoints:
    def __init__(self, root, identity, disk_bytes=10_000_000_000, external_bytes=0):
        """Keep attempt identity; create files only when operations need them."""
        self.root = Path(root)
        self.identity = identity
        self.disk_bytes = disk_bytes
        self.external_bytes = external_bytes
        self.recovery = None
        self._locked = False

    @contextmanager
    def lock(self):
        """Allow one process to publish or restore this attempt at a time.
        Nested calls on this same manager reuse its lock, as run invokes save/load internally.
        Always release the OS lock when the outer operation exits.
        """
        if self._locked:
            yield
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / ".writer.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._locked = True
            try:
                yield
            finally:
                self._locked = False
                fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, additional):
        """Charge retained artifacts and offload files before allocating a staging checkpoint."""
        used = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        external = self.external_bytes() if callable(self.external_bytes) else self.external_bytes
        if used + external + additional > self.disk_bytes:
            raise OSError("Run storage budget cannot hold checkpoint staging")
        if shutil.disk_usage(self.root).free < additional:
            raise OSError("Insufficient free disk space for checkpoint staging")

    def blob(self, model):
        """Save a frozen teacher separately from mutable recovery checkpoints.
        Publish its completion manifest only after the adapter output is flushed.
        Remove an incomplete snapshot on failure; return the run-relative reference on success.
        """
        self.reserve(model.info.checkpoint_bytes + 65536)
        relative = f"teachers/{uuid.uuid4().hex}"
        path = self.root / relative
        path.mkdir(parents=True)
        try:
            model.save(path, "frozen")
            synchronize(path)
            write_json(path / "complete.json", {"files": checksums(path)})
            self.reserve(0)
        except BaseException:
            shutil.rmtree(path)
            raise
        return relative

    def publish(self, state, learner, auxiliary, dependencies, *, completed=False):
        """Stage policy state, learner state and any active auxiliary as one recovery unit.
        Replace the latest pointer after the staged unit is flushed.
        Retain the new checkpoint plus one fallback while running, or only the final checkpoint.
        Prune obsolete payloads only after the new recovery unit has been published.
        """
        encoded_size = len(json.dumps(state, allow_nan=False).encode())
        estimate = learner.info.checkpoint_bytes + encoded_size + 65536
        if auxiliary is not None:
            estimate += auxiliary.info.checkpoint_bytes
        self.reserve(estimate)
        folder = self.root / "checkpoints"
        folder.mkdir(exist_ok=True)
        name = uuid.uuid4().hex
        stage = folder / (".stage-" + name)
        stage.mkdir()
        try:
            learner_path = stage / "learner"
            learner_path.mkdir()
            learner.save(learner_path, "resume")
            if auxiliary is not None:
                aux_path = stage / "auxiliary"
                aux_path.mkdir()
                auxiliary.save(aux_path, "resume")
            write_json(stage / "state.json", state)
            synchronize(stage)
            manifest = {"identity": self.identity, "dependencies": sorted(set(dependencies)),
                        "files": checksums(stage), "event": state["event"]}
            write_json(stage / "complete.json", manifest)
            if checksums(stage) != manifest["files"]:
                raise ValueError("Checkpoint verification failed")
            self.reserve(0)
            os.replace(stage, folder / name)
            descriptor = os.open(folder, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            latest_path = self.root / "latest.json"
            previous = read_json(latest_path)["checkpoints"] if latest_path.exists() else []
            # A fallback recovery must never retain the corrupt newer checkpoint.
            if self.recovery:
                previous = [self.recovery["restored"]]
            retained = [name] if completed else [name, *previous[:1]]
            write_json(latest_path, {"checkpoints": retained})
            self.recovery = None
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise
        self.prune(retained)
        return str(folder / name)

    def prune(self, retained):
        """Find teacher blobs referenced by every retained recovery checkpoint.
        Remove other model/replay payloads and abandoned staging directories.
        Keep small audit manifests; teachers survive until neither retained checkpoint needs them.
        """
        live_blobs = set()
        folder = self.root / "checkpoints"
        for name in retained:
            live_blobs.update(read_json(folder / name / "complete.json")["dependencies"])
        for path in folder.iterdir():
            if path.is_dir() and path.name not in retained:
                # Keep small manifests for audit; remove only owned numerical state.
                for name in ("learner", "auxiliary"):
                    if (path / name).exists():
                        shutil.rmtree(path / name)
                # state.json contains replay payload, not merely audit metadata.
                if (path / "state.json").exists():
                    (path / "state.json").unlink()
                if path.name.startswith(".stage-"):
                    shutil.rmtree(path)
        teacher_root = self.root / "teachers"
        if teacher_root.exists():
            for path in teacher_root.iterdir():
                if str(path.relative_to(self.root)) not in live_blobs:
                    shutil.rmtree(path)

    def latest(self):
        """Try the latest checkpoint, then its retained fallback, as complete recovery units.
        Require matching run identity and file checksums.
        Record a fallback in recovery.json so discarded progress is visible to the caller.
        """
        pointer = self.root / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError("This attempt has no complete recovery checkpoint")
        failures = []
        for name in read_json(pointer)["checkpoints"]:
            try:
                path = self.root / "checkpoints" / name
                manifest = read_json(path / "complete.json")
                if manifest["identity"] != self.identity:
                    raise ValueError("Scientific configuration or implementation changed")
                if checksums(path) != manifest["files"]:
                    raise ValueError("Checkpoint checksum mismatch")
                if failures:
                    self.recovery = {"restored": name, "invalid": failures,
                                     "lost_work": "Uncheckpointed work is unmeasured"}
                    write_json(self.root / "recovery.json", self.recovery)
                return path, read_json(path / "state.json")
            except (OSError, ValueError, KeyError) as error:
                failures.append({"checkpoint": name, "reason": str(error)})
        raise ValueError(f"No compatible, complete recovery checkpoint: {failures}")


class ReplayStore:
    """Materialized active membership. Only the lifecycle owner replaces it."""

    def __init__(self, records=(), *, budget):
        """Materialize the active, unique train-only replay membership.
        Count examples and tokens, and use the larger of declared or serialized payload bytes.
        Check every budget independently; selection and eviction remain pipeline decisions.
        """
        self.records = training_records(records)
        self.budget = budget
        declared_bytes = sum(r.storage_bytes for r in self.records)
        serialized_bytes = len(json.dumps(self.state_dict(), allow_nan=False).encode()) if self.records else 0
        self.usage = {"examples": len(self.records), "tokens": sum(r.tokens for r in self.records),
                      "bytes": max(declared_bytes, serialized_bytes)}
        if budget.examples is not None and len(self.records) > budget.examples:
            raise ValueError("Replay example budget exceeded")
        if self.usage["tokens"] > budget.tokens:
            raise ValueError("Replay token budget exceeded")
        if self.usage["bytes"] > budget.bytes:
            raise ValueError("Replay payload byte budget exceeded")

    def task(self, task_id):
        """Return only the retained records belonging to the requested task."""
        return tuple(r for r in self.records if r.task_id == task_id)

    @property
    def tasks(self):
        """Expose sorted task IDs with nonempty retained membership for replay routing."""
        return tuple(sorted({r.task_id for r in self.records}))

    def proposal(self, records):
        """Build a budget-checked replacement without mutating the currently active store."""
        return ReplayStore(records, budget=self.budget)

    def state_dict(self):
        """Serialize retained record payloads as plain data for full-run checkpoints."""
        return [asdict(r) for r in self.records]

    @classmethod
    def restore(cls, state, budget):
        """Rebuild replay records from saved state and reapply the supplied storage limits."""
        return cls((Record(**r) for r in state), budget=budget)


def tuple_tree(value):
    """Undo JSON list conversion for the nested tuple structure required by Random.setstate."""
    return tuple(tuple_tree(v) for v in value) if isinstance(value, list) else value


def save_weights(model, path):
    """Delegate a standalone frozen-weight snapshot to the model adapter.
    This helper does not save pipeline cursors, replay membership or optimizer state.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save(path, "frozen")
    return path


def base_assets(path):
    """Hash immutable model-cache assets, following content-addressed cache symlinks."""
    result = {}
    for file in sorted(Path(path).rglob("*")):
        if file.is_file():
            with file.open("rb") as stream:
                result[str(file.relative_to(path))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def tensor_fingerprint(value):
    """Hash Torch serialization as a stream instead of collecting a second checkpoint in RAM."""
    import torch
    digest = hashlib.sha256()
    class DigestWriter:
        def write(self, data):
            digest.update(data)
            return len(data)
        def flush(self):
            pass
    torch.save(value, DigestWriter())
    return digest.hexdigest()
