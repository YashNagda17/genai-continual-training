"""The blind manual check: two raters score 30 test summaries per language for faithfulness and fluency (1-3).

After the raters fill the sheet: python -m src.new.phases.manual_check RESULTS/phase7"""

from collections import Counter
from pathlib import Path
import csv
import random
import statistics
import sys

from ..datasets.dataset_instruction import LANGUAGES
from ..utils.numerics import read_json, write_json
from .support import screening

CRITERIA = ("faithfulness", "fluency")
FIELDS = tuple(f"{criterion}_{rater}" for rater in (1, 2) for criterion in CRITERIA)


def rating_sheet(directory, dataset, arms, count=30, seed=0):
    """The same articles for every arm, shuffled, with the arm hidden. A separate key maps rows to arms."""
    sheet, key = Path(directory) / "manual_ratings.csv", Path(directory) / "manual_key.json"
    if not sheet.is_file():
        rng, rows = random.Random(seed), []
        for language in LANGUAGES:
            articles = {r.id: r for r in dataset.view(f"{language}/test")}
            ids = rng.sample(sorted(articles), min(count, len(articles)))
            for name, arm in arms.items():
                written = {row["id"]: row["prediction"] for row in read_json(
                    Path(screening(arm)["attempt"]) / "predictions" / f"{language}-test-plain.json")}
                rows += [(name, language, i, articles[i].text, written[i]) for i in ids]
        rng.shuffle(rows)
        with sheet.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("row", "language", "article", "summary", *FIELDS))
            writer.writerows((index, language, text, summary, *[""] * len(FIELDS))
                             for index, (_, language, _, text, summary) in enumerate(rows))
        write_json(key, {str(index): {"arm": name, "id": i} for index, (name, _, i, _, _) in enumerate(rows)})
    return agreement(sheet, key)


def kappa(left, right):
    """Cohen's kappa between the two raters."""
    a, b = Counter(left), Counter(right)
    observed = sum(x == y for x, y in zip(left, right)) / len(left)
    chance = sum(a[k] * b[k] for k in set(a) | set(b)) / len(left) ** 2
    return (observed - chance) / (1 - chance) if chance < 1 else None


def agreement(sheet, key):
    with Path(sheet).open(encoding="utf-8") as handle:
        rated = [row for row in csv.DictReader(handle) if all(row[f].strip() for f in FIELDS)]
    if not rated:
        return {"status": "awaiting ratings", "sheet": str(sheet)}
    arms, result = read_json(key), {"status": "rated", "rows": len(rated)}
    for criterion in CRITERIA:
        first, second = ([int(row[f"{criterion}_{rater}"]) for row in rated] for rater in (1, 2))
        by_arm = {}
        for row, x, y in zip(rated, first, second):
            by_arm.setdefault(f"{arms[row['row']]['arm']}/{row['language']}", []).append((x + y) / 2)
        result[criterion] = {"kappa": kappa(first, second),
                             "mean_by_arm": {name: statistics.mean(v) for name, v in sorted(by_arm.items())}}
    return result


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    print(agreement(folder / "manual_ratings.csv", folder / "manual_key.json"))
