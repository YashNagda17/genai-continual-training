"""Replay membership, weight snapshots, and checksummed checkpoint storage."""

from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import time
import uuid

if os.name == "nt":
    import msvcrt
else:
    import fcntl

from .contracts import Record, training_records


def is_folder_metadata(path):
    """Windows Explorer files are not model or recovery payloads."""
    return Path(path).name.casefold() in {"desktop.ini", "thumbs.db"}


def replace_artifact(source, destination):
    """Preserve atomic publication, retrying transient Windows permission failures."""
    for attempt in range(5):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 4:
                raise
            time.sleep(.1 * 2 ** attempt)


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
    replace_artifact(temporary, path)
    _fsync_directory(path.parent)


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
        if is_folder_metadata(file):
            continue
        if file.is_symlink():
            raise ValueError("Checkpoint files must not be symlinks")
        if file.is_file() and file != Path(path) / "complete.json":
            with file.open("rb") as stream:
                result[str(file.relative_to(path))] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def _fsync_directory(path):
    """Flush a directory entry when the platform allows it.

    Windows can reject a directory handle. The file contents were already fsynced.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _lock_handle(handle):
    """Take one exclusive non-blocking writer lock.

    POSIX keeps fcntl.flock. Windows locks a single byte because fcntl is absent.
    """
    if os.name == "nt":
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_handle(handle):
    """Release the lock taken by _lock_handle."""
    if os.name == "nt":
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(handle, fcntl.LOCK_UN)


def _fsync_file(path):
    """Flush one file. Windows rejects fsync on a read-only handle, so retry read/write."""
    for flag in (os.O_RDONLY, os.O_RDWR):
        try:
            descriptor = os.open(path, flag)
        except OSError:
            continue
        try:
            os.fsync(descriptor)
            return
        except OSError:
            continue
        finally:
            os.close(descriptor)


def synchronize(path):
    """Flush model files and their directory before writing the completion marker."""
    for file in Path(path).rglob("*"):
        if is_folder_metadata(file):
            continue
        if file.is_file():
            _fsync_file(file)
    _fsync_directory(path)


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
        with (self.root / ".writer.lock").open("a+b") as lock:
            _lock_handle(lock)
            self._locked = True
            try:
                yield
            finally:
                self._locked = False
                _unlock_handle(lock)

    def reserve(self, additional):
        """Charge retained artifacts and offload files before allocating a staging checkpoint."""
        used, pending = 0, [self.root]
        while pending:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(entry.path)
                    elif not is_folder_metadata(entry.name) and entry.is_file():
                        used += entry.stat().st_size
        external = self.external_bytes() if callable(self.external_bytes) else self.external_bytes
        if used + external + additional > self.disk_bytes:
            raise OSError("Run storage budget cannot hold checkpoint staging")
        if shutil.disk_usage(self.root).free < additional:
            raise OSError("Insufficient free disk space for checkpoint staging")

    def pin(self, model, record):
        """Publish a complete immutable English checkpoint for one prefix variant."""
        if record.get("kind") != "english" or record.get("writer") != "english_checkpoint":
            raise ValueError("Only the English stage can pin the English checkpoint")
        prefix = str(record["prefix"])
        if prefix in {"", ".", ".."} or "/" in prefix or "\\" in prefix:
            raise ValueError("Prefix artifact names must be single path components")
        destination = self.root / "pinned" / "english" / prefix
        return self.pin_artifact(destination, model.info.checkpoint_bytes + 65536,
                                  lambda path: model.save(path, "english", record=record))

    def pin_pool(self, records, *, english_checkpoint):
        """Publish the pool once; committed answers are never regenerated or replaced."""
        rows = [dict(row) for row in records]
        if any(not {"id", "summary", "baseline", "loss_before"} <= set(row) for row in rows):
            raise ValueError("Pool record needs a summary, a baseline, and loss_before")
        payload = {"english_checkpoint": english_checkpoint, "writer": "english_checkpoint", "records": rows}
        return self.pin_artifact(self.root / "pinned" / "pool", len(json.dumps(payload).encode()) + 65536,
                                  lambda path: write_json(path / "manifest.json", payload))

    def pin_artifact(self, destination, estimate, writer):
        """Atomically publish one immutable run-owned artifact using the supplied writer."""
        destination = Path(destination)
        if destination.resolve() == self.root.resolve() or not destination.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Pinned artifacts must stay inside their attempt')
        with self.lock():
            complete = destination / "complete.json"
            if complete.is_file():
                if checksums(destination) != read_json(complete)["files"]:
                    raise ValueError("Pinned artifact is corrupt")
                return destination.relative_to(self.root).as_posix()
            if destination.exists():
                raise ValueError("Pinned artifact is incomplete; preserve it for recovery inspection")
            destination.parent.mkdir(parents=True, exist_ok=True)
            for abandoned in destination.parent.glob(f".stage-pin-{destination.name}-*"):
                self._remove_owned(abandoned)
            self.reserve(estimate)
            stage = destination.with_name(f".stage-pin-{destination.name}-{uuid.uuid4().hex}")
            stage.mkdir()
            try:
                writer(stage)
                synchronize(stage)
                write_json(stage / "complete.json", {"files": checksums(stage)})
                self.reserve(0)
                replace_artifact(stage, destination)
                _fsync_directory(destination.parent)
            except BaseException:
                if stage.exists():
                    self._remove_owned(stage)
                raise
            return destination.relative_to(self.root).as_posix()

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
            replace_artifact(stage, folder / name)
            _fsync_directory(folder)
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

    def _remove_owned(self, path):
        """Remove obsolete run-owned trees, handling Windows read-only attributes and locks."""
        path = Path(path)
        root = self.root.resolve()
        target = path.resolve()
        if target == root or not target.is_relative_to(root):
            raise ValueError("Cleanup target must remain inside the attempt directory")

        def retry_readonly(operation, failed_path, error):
            resolved = Path(failed_path).resolve()
            if resolved != target and not resolved.is_relative_to(target):
                raise ValueError("Cleanup callback escaped the obsolete artifact")
            if isinstance(error[1], FileNotFoundError):
                return  # Another actor already removed this obsolete entry.
            # Windows directory READONLY attributes can prevent rmdir after files are gone.
            if os.name != "nt" or not isinstance(error[1], PermissionError):
                raise error[1]
            os.chmod(failed_path, os.stat(failed_path).st_mode | stat.S_IWRITE)
            operation(failed_path)

        for attempt in range(3):
            try:
                shutil.rmtree(path, onerror=retry_readonly)
                return
            except FileNotFoundError:
                if not path.exists():
                    return
                raise
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(.1 * (attempt + 1))

    def prune(self, retained):
        """Find teacher blobs referenced by every retained recovery checkpoint.
        Remove other model/replay payloads and abandoned staging directories.
        Keep small audit manifests; teachers survive until neither retained checkpoint needs them.
        Pinned English checkpoints and pools live under pinned/ and are never removed here.
        """
        live_blobs = set()
        folder = self.root / "checkpoints"
        if folder.exists():
            for name in retained:
                live_blobs.update(read_json(folder / name / "complete.json")["dependencies"])
        for path in folder.iterdir() if folder.exists() else ():
            if path.is_dir() and path.name not in retained:
                # Keep small manifests for audit; remove only owned numerical state.
                for name in ("learner", "auxiliary"):
                    if (path / name).exists():
                        self._remove_owned(path / name)
                # state.json contains replay payload, not merely audit metadata.
                if (path / "state.json").exists():
                    (path / "state.json").unlink()
                if path.name.startswith(".stage-"):
                    self._remove_owned(path)
        teacher_root = self.root / "teachers"
        if teacher_root.exists():
            for path in teacher_root.iterdir():
                if path.relative_to(self.root).as_posix() not in live_blobs:
                    self._remove_owned(path)

    def latest(self, *, verify=True):
        """Try the latest checkpoint, then its retained fallback, as complete recovery units.
        By default require matching run identity and file checksums; verify=False loads the newest directly.
        Record a fallback in recovery.json so discarded progress is visible to the caller.
        """
        pointer = self.root / "latest.json"
        if not pointer.exists():
            raise FileNotFoundError("This attempt has no complete recovery checkpoint")
        if not verify:
            checkpoints = read_json(pointer)["checkpoints"]
            if not checkpoints:
                raise FileNotFoundError("This attempt has no recovery checkpoint")
            path = self.root / "checkpoints" / checkpoints[0]
            self.recovery = None
            return path, read_json(path / "state.json")
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
                      "answer_tokens": sum(len(r.practice.get('answer_ids', ())) for r in self.records if r.practice),
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

