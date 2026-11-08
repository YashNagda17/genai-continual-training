"""Fetch CLINC, Schuster, Bitext, TRACE, and XL-Sum into canonical JSONL."""

from pathlib import Path
import json
import random
import tarfile
import zipfile

from .dataset_instruction import InstructionDataset, SummarizationDataset

SPLITS = ("train", "validation", "test")
TRACE_TASKS = ("C-STANCE", "FOMC", "MeetingBank", "Py150", "ScienceQA",
               "NumGLUE-cm", "NumGLUE-ds", "20Minuten")
TRACE_CHOICE = {"FOMC", "C-STANCE"}
TRACE_ANSWER = {"NumGLUE-cm", "NumGLUE-ds"}
SPLIT_NAMES = {"train": "train", "eval": "validation", "validation": "validation", "test": "test"}


def canonical(source_id, task, split, prompt, response, label, group=None):
    """One instruction row. A missing class stays JSON null and is not invented from the task."""
    row = {"id": source_id, "task": task, "split": split, "prompt": prompt,
           "response": response, "class": label}
    if group is not None:
        row["group"] = group
    return row


def clinc_rows(by_split, names):
    """Map plus-split texts and intent indexes to one task whose response is the intent name."""
    rows = []
    for split in SPLITS:
        for index, (text, intent) in enumerate(by_split[split]):
            label = names[int(intent)]
            prompt = str(text).strip()
            if prompt:
                rows.append(canonical(f"{split}-{index:05d}", "clinc", split, prompt, label, label))
    return rows


# Official order and membership from clinc/oos-eval data/domains.json.
CLINC_DOMAINS = {
    "banking": ("freeze_account", "routing", "pin_change", "bill_due", "pay_bill",
                "account_blocked", "interest_rate", "min_payment", "bill_balance", "transfer",
                "order_checks", "balance", "spending_history", "transactions", "report_fraud"),
    "credit_cards": ("replacement_card_duration", "expiration_date", "damaged_card",
                     "improve_credit_score", "report_lost_card", "card_declined",
                     "credit_limit_change", "apr", "redeem_rewards", "credit_limit",
                     "rewards_balance", "application_status", "credit_score", "new_card",
                     "international_fees"),
    "kitchen_and_dining": ("food_last", "confirm_reservation", "how_busy", "ingredients_list",
                           "calories", "nutrition_info", "recipe", "restaurant_reviews",
                           "restaurant_reservation", "meal_suggestion", "restaurant_suggestion",
                           "cancel_reservation", "ingredient_substitution", "cook_time",
                           "accept_reservations"),
    "home": ("what_song", "play_music", "todo_list_update", "reminder", "reminder_update",
             "calendar_update", "order_status", "update_playlist", "shopping_list", "calendar",
             "next_song", "order", "todo_list", "shopping_list_update", "smart_home"),
    "auto_and_commute": ("current_location", "oil_change_when", "oil_change_how", "uber",
                         "traffic", "tire_pressure", "schedule_maintenance", "gas", "mpg",
                         "distance", "directions", "last_maintenance", "gas_type", "tire_change",
                         "jump_start"),
    "travel": ("plug_type", "travel_notification", "translate", "flight_status",
               "international_visa", "timezone", "exchange_rate", "travel_suggestion",
               "travel_alert", "vaccines", "lost_luggage", "book_flight", "book_hotel",
               "carry_on", "car_rental"),
    "utility": ("weather", "alarm", "date", "find_phone", "share_location", "timer",
                "make_call", "calculator", "definition", "measurement_conversion", "flip_coin",
                "spelling", "time", "roll_dice", "text"),
    "work": ("pto_request_status", "next_holiday", "insurance_change", "insurance",
             "meeting_schedule", "payday", "taxes", "income", "rollover_401k", "pto_balance",
             "pto_request", "w2", "schedule_meeting", "direct_deposit", "pto_used"),
    "small_talk": ("who_made_you", "meaning_of_life", "who_do_you_work_for", "do_you_have_pets",
                   "what_are_your_hobbies", "fun_fact", "what_is_your_name", "where_are_you_from",
                   "goodbye", "thank_you", "greeting", "tell_joke", "are_you_a_bot",
                   "how_old_are_you", "what_can_i_ask_you"),
    "meta": ("change_speed", "user_name", "whisper_mode", "yes", "change_volume", "no",
             "change_language", "repeat", "change_accent", "cancel", "sync_device",
             "change_user_name", "change_ai_name", "reset_settings", "maybe"),
}


def clinc_domain_of(intent):
    """Return the official CLINC domain for one in-scope intent name."""
    for domain, intents in CLINC_DOMAINS.items():
        if intent in intents:
            return domain
    return None


def clinc_domain_rows(by_split, names):
    """Map plus-split rows onto the ten official domains and count omitted oos rows."""
    official = {intent for intents in CLINC_DOMAINS.values() for intent in intents}
    if any(len(intents) != 15 for intents in CLINC_DOMAINS.values()) or len(official) != 150:
        raise ValueError("CLINC domain map must contain 10 domains of 15 unique intents")
    unknown = sorted({name for name in names if name != "oos" and name not in official})
    missing = sorted(official.difference(names))
    if unknown or missing:
        raise ValueError(f"CLINC intent schema does not match domains.json: unknown={unknown} missing={missing}")
    rows, dropped = [], 0
    for split in SPLITS:
        for index, (text, intent) in enumerate(by_split[split]):
            label = names[int(intent)]
            prompt = str(text).strip()
            if not prompt:
                continue
            if label == "oos":
                dropped += 1
                continue
            rows.append(canonical(f"{split}-{index:05d}", clinc_domain_of(label), split, prompt, label, label))
    present = {row["task"] for row in rows if row["split"] == "train"}
    absent = [domain for domain in CLINC_DOMAINS if domain not in present]
    if absent:
        raise ValueError(f"CLINC domain stream is missing training rows for {absent}")
    return rows, {"oos_dropped": dropped,
                  "task_order": [domain for domain in CLINC_DOMAINS if domain in {row["task"] for row in rows}]}


def schuster_rows(by_split):
    """Map English intent rows. The task is the domain prefix; slots are omitted."""
    rows = []
    for raw_split in ("train", "eval", "validation", "test"):
        if raw_split not in by_split:
            continue
        split = SPLIT_NAMES[raw_split]
        for index, row in enumerate(by_split[raw_split]):
            intent = str(row["intent_label"]).strip()
            prompt = str(row["untokenized_utterance"]).strip()
            if intent and prompt:
                rows.append(canonical(f"{split}-{index:05d}", intent.split("/", 1)[0],
                                      split, prompt, intent, intent))
    return rows


def bitext_rows(records, seed=0):
    """Assign a seeded group holdout because the source file is one train split."""
    groups = {}
    for index, row in enumerate(records):
        category = str(row["category"]).strip()
        instruction = str(row["instruction"]).strip()
        groups.setdefault(category, {}).setdefault(instruction, []).append(index)
    chosen = {}
    for category, members in groups.items():
        names = sorted(members)
        random.Random(f"{seed}:{category}").shuffle(names)
        n_test = int(len(names) * 0.1)
        n_val = int(len(names) * 0.1)
        test, validation = set(names[:n_test]), set(names[n_test:n_test + n_val])
        for name in names:
            split = "test" if name in test else "validation" if name in validation else "train"
            for index in members[name]:
                chosen[index] = split
    rows = []
    for index, row in enumerate(records):
        prompt = str(row["instruction"]).strip()
        response = str(row["response"]).strip()
        label = str(row["intent"]).strip()
        if prompt and response and label:
            rows.append(canonical(f"{index:05d}", str(row["category"]).strip(), chosen[index],
                                  prompt, response, label,
                                  prompt))
    return rows


def trace_class(task, answer):
    """Return a genuine label. Generation tasks and unlabeled ScienceQA answers stay unset."""
    text = answer.strip()
    if task in TRACE_CHOICE or task in TRACE_ANSWER:
        return text
    if task == "ScienceQA" and text[:1] in "ABCD" and (len(text) == 1 or text[1] == "\n"):
        return text[0]
    return None


XLSUM_LANGUAGES = ("english", "hindi")
XLSUM_SPLIT_FILES = (("train", "train"), ("validation", "val"), ("test", "test"))


def xlsum_archive_rows(archive_path, language):
    """Read one language tar. Official `val` is stored as `validation`."""
    rows, dropped, seen = [], 0, set()
    with tarfile.open(archive_path, "r:bz2") as archive:
        members = {Path(member.name).name: member for member in archive.getmembers() if member.isfile()}
        for split, suffix in XLSUM_SPLIT_FILES:
            name = f"{language}_{suffix}.jsonl"
            member = members.get(name)
            if member is None:
                raise ValueError(f"{archive_path} is missing {name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"{archive_path} could not read {name}")
            for line in handle:
                data = json.loads(line)
                article_id = str(data.get("id", "")).strip()
                article = str(data.get("text", "")).strip()
                summary = str(data.get("summary", "")).strip()
                if not article_id or not article or not summary:
                    dropped += 1
                    continue
                row_id = f"{language}/{article_id}"
                if row_id in seen:
                    raise ValueError(f"Duplicate XL-Sum id {row_id}")
                seen.add(row_id)
                rows.append({"id": row_id, "task": language, "split": split, "language": language,
                             "text": article, "summary": summary})
    return rows, dropped


def trace_rows(tasks):
    """Convert the eight TRACE-5000 tasks. Lima is not a task in this stream."""
    rows = []
    for task in TRACE_TASKS:
        files = tasks[task]
        for raw_split, split in (("train", "train"), ("eval", "validation"), ("test", "test")):
            for index, row in enumerate(files[raw_split]):
                prompt = str(row["prompt"]).strip()
                answer = row.get("answer")
                if prompt and isinstance(answer, str) and answer.strip():
                    rows.append(canonical(f"{task}-{split}-{index:05d}", task, split,
                                          prompt, answer, trace_class(task, answer)))
    return rows


def publish(root, dataset_id, source, revision, rows, extra):
    """Write canonical JSONL and its manifest. An existing directory is kept."""
    counts = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    if counts["train"] == 0:
        raise ValueError(f"{dataset_id} produced no training records")
    task_counts = {}
    for row in rows:
        bucket = task_counts.setdefault(row["task"], {split: 0 for split in SPLITS})
        bucket[row["split"]] += 1
    out = Path(root) / dataset_id / revision
    records = out / "records.jsonl"
    manifest_path = out / "manifest.json"
    if records.exists() and manifest_path.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    with records.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"dataset_id": dataset_id, "source": source, "revision": revision,
                "counts": counts, "task_counts": task_counts,
                "task_order": list(dict.fromkeys(row["task"] for row in rows)),
                "records": len(rows), **extra}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_source(directory, dataset_cls):
    """Load a fetched directory into the given dataset class. This does not download."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    dataset = dataset_cls()
    raw, _ = dataset.load({"path": directory / "records.jsonl", "dataset_id": manifest["dataset_id"],
                           "revision": manifest.get("revision", manifest["dataset_id"])})
    dataset.manifest["source"] = dict(manifest)
    return dataset, raw


class InstructionSource:
    """Download one Hub snapshot and convert it. Subclasses own the file layout."""

    dataset_id = ""
    source = ""
    patterns = ()
    extra = {}

    def download(self, root, token=None, revision="main"):
        from huggingface_hub import snapshot_download
        cache = Path(root) / self.dataset_id / "hub"
        cache.mkdir(parents=True, exist_ok=True)
        kwargs = {"repo_id": self.source, "repo_type": "dataset", "revision": revision,
                  "cache_dir": str(cache), "allow_patterns": list(self.patterns)}
        if token:
            kwargs["token"] = token
        return Path(snapshot_download(**kwargs))

    def convert(self, snapshot):
        raise NotImplementedError

    def fetch(self, root, token=None, revision="main"):
        snapshot = self.download(root, token, revision)
        rows = self.convert(snapshot)
        return publish(root, self.dataset_id, self.source, snapshot.name,
                       rows, {**self.extra, **getattr(self, "report", {})})


class ClincOOSSource(InstructionSource):
    dataset_id = "clinc_oos"
    source = "clinc/clinc_oos"
    patterns = ("plus/*.parquet", "README.md")
    extra = {"config": "plus", "protocol": "official", "truncation": "drop",
             "usage": "Official plus splits, including out-of-scope, as one task."}

    def _read(self, snapshot):
        import pyarrow.parquet as pq
        by_split, names = {}, None
        for split in SPLITS:
            table = pq.read_table(snapshot / "plus" / f"{split}-00000-of-00001.parquet")
            if names is None:
                meta = json.loads(table.schema.metadata[b"huggingface"])
                names = meta["info"]["features"]["intent"]["names"]
            data = table.to_pydict()
            by_split[split] = list(zip(data["text"], data["intent"]))
        return by_split, names

    def convert(self, snapshot):
        return clinc_rows(*self._read(snapshot))


class ClincDomainSource(ClincOOSSource):
    """Same plus snapshot as the one-task CLINC file, with one task per official domain."""

    dataset_id = "clinc_oos_domains"
    extra = {"config": "plus", "protocol": "official-domains", "truncation": "drop",
             "domain_map": "clinc/oos-eval data/domains.json",
             "usage": "Ten official domains are tasks. Out-of-scope rows are omitted and counted."}

    def convert(self, snapshot):
        rows, self.report = clinc_domain_rows(*self._read(snapshot))
        return rows



class SchusterSource(InstructionSource):
    dataset_id = "schuster"
    source = "edmdias/multilingual-task-oriented-dialog"
    patterns = ("en/*.parquet", "README.md")
    extra = {"config": "en", "protocol": "official", "truncation": "drop", "license": "cc-by-sa-4.0",
             "usage": "English intents only. Hub eval is validation. Slots are not supervised."}

    def convert(self, snapshot):
        import pyarrow.parquet as pq
        files = {"train": "train-en.parquet", "eval": "eval-en.parquet", "test": "test-en.parquet"}
        by_split = {split: pq.read_table(snapshot / "en" / name).to_pylist()
                    for split, name in files.items()}
        return schuster_rows(by_split)


class BitextSource(InstructionSource):
    dataset_id = "bitext_customer_support"
    source = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    patterns = ("*.csv", "README.md")
    extra = {"protocol": "category-instruction-groups-seed-0-10-10", "truncation": "prompt_left",
             "license": "cdla-sharing-1.0",
             "usage": "Synthetic customer-support replies. Split is the seeded group holdout."}

    def convert(self, snapshot):
        import csv
        files = [path for path in snapshot.glob("*.csv") if path.is_file()]
        if len(files) != 1:
            raise ValueError(f"Expected one CSV in {snapshot}, found {len(files)}")
        with files[0].open(newline="", encoding="utf-8") as handle:
            return bitext_rows(list(csv.DictReader(handle)))


class XLSumSource(InstructionSource):
    """English and Hindi XL-Sum. Marathi and Gujarati stay out until Phase 7."""

    dataset_id = "xlsum"
    source = "csebuetnlp/xlsum"
    patterns = tuple(f"data/{language}_XLSum_v2.0.tar.bz2" for language in XLSUM_LANGUAGES)
    extra = {"protocol": "official", "license": "cc-by-nc-sa-4.0", "schema": "summarization",
             "languages": list(XLSUM_LANGUAGES),
             "usage": "Official splits. The article is the input and the summary is the target."}

    def convert(self, snapshot):
        rows, dropped = [], 0
        for language in XLSUM_LANGUAGES:
            archive = snapshot / "data" / f"{language}_XLSum_v2.0.tar.bz2"
            if not archive.is_file():
                raise ValueError(f"XL-Sum snapshot is missing {archive.name}")
            language_rows, language_dropped = xlsum_archive_rows(archive, language)
            rows.extend(language_rows)
            dropped += language_dropped
        self.report = {"empty_dropped": dropped}
        return rows



class TraceSource(InstructionSource):
    dataset_id = "trace"
    source = "YeMoKoo/TRACE"
    patterns = ("TRACE-Benchmark.zip",)
    extra = {"config": "LLM-CL-Benchmark_5000", "protocol": "official", "truncation": "prompt_left",
             "usage": "Eight TRACE tasks. Lima is excluded. eval.json is validation."}

    def convert(self, snapshot):
        with zipfile.ZipFile(snapshot / "TRACE-Benchmark.zip") as archive:
            tasks = {}
            for task in TRACE_TASKS:
                tasks[task] = {}
                for name in ("train.json", "eval.json", "test.json"):
                    member = f"TRACE-Benchmark/LLM-CL-Benchmark_5000/{task}/{name}"
                    key = "eval" if name == "eval.json" else name[:-5]
                    try:
                        tasks[task][key] = json.loads(archive.read(member))
                    except KeyError as error:
                        raise ValueError(f"TRACE archive is missing {member}") from error
        return trace_rows(tasks)


SOURCES = {"clinc_oos": ClincOOSSource, "clinc_oos_domains": ClincDomainSource,
           "schuster": SchusterSource, "bitext_customer_support": BitextSource, "trace": TraceSource,
           "xlsum": XLSumSource}


def fetch_source(name, root, token=None, revision="main"):
    """Fetch one named source, or every source when name is 'all' is handled by the CLI."""
    try:
        source = SOURCES[name]()
    except KeyError as error:
        raise ValueError(f"Unknown instruction source {name}") from error
    return source.fetch(root, token=token, revision=revision)


def main(argv):
    name = argv[1]
    root = Path(argv[2]) if len(argv) > 2 else Path("src/datasets/data")
    token = argv[3] if len(argv) > 3 and argv[3] else None
    names = list(SOURCES) if name == "all" else [name]
    for item in names:
        print(fetch_source(item, root, token=token))


if __name__ == "__main__":
    import sys
    main(sys.argv)
