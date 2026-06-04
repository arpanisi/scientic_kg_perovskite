import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

from src.data import build_datasets


def test_builds_grouped_train_validation_test_jsonl(tmp_path, capsys):
    csv_path = tmp_path / "source.csv"
    output_dir = tmp_path / "out"

    split_keys = one_key_per_split(seed=17)
    write_source_csv(csv_path, split_keys)

    argv = [
        "build_datasets.py",
        "--csv",
        str(csv_path),
        "--output-dir",
        str(output_dir),
        "--input-representation",
        "core",
        "--output-schema",
        "performance_only",
    ]
    with patch.object(sys, "argv", argv):
        build_datasets.main()
    capsys.readouterr()

    train_path = output_dir / "core__performance_only.train.jsonl"
    validation_path = output_dir / "core__performance_only.validation.jsonl"
    test_path = output_dir / "core__performance_only.test.jsonl"
    summary_path = output_dir / "core__performance_only.summary.json"

    assert train_path.exists()
    assert validation_path.exists()
    assert test_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["rows_read"] == 3
    assert summary["rows_written"] == 3
    assert summary["rows_skipped"] == 0
    assert summary["split_counts"] == {"train": 1, "validation": 1, "test": 1}

    record = json.loads(train_path.read_text(encoding="utf-8").splitlines()[0])
    assert [message["role"] for message in record["messages"]] == ["system", "user", "assistant"]

    user_prompt = record["messages"][1]["content"]
    assert "Task: Predict perovskite solar-cell JV performance." in user_prompt
    assert "cell architecture: nip" in user_prompt
    assert "perovskite composition long form: MAPbI3" in user_prompt
    assert "jv default pce:" not in user_prompt.lower()

    assistant_payload = json.loads(record["messages"][2]["content"])
    assert assistant_payload["prediction"]["pce_bin"] == "10-15"
    assert assistant_payload["prediction"]["voc_bin"] == "1-1.1"
    assert assistant_payload["prediction"]["jsc_bin"] == "20-25"
    assert assistant_payload["prediction"]["ff_bin"] == "0.6-0.7"


def one_key_per_split(seed: int) -> dict[str, str]:
    found = {}
    for index in range(1000):
        key = f"10.0000/test-{index}"
        split = build_datasets.split_name_for_key(key, 0.80, 0.10, seed)
        found.setdefault(split, key)
        if set(found) == {"train", "validation", "test"}:
            return found
    raise AssertionError("Could not find deterministic split keys for test setup.")


def write_source_csv(csv_path: Path, split_keys: dict[str, str]) -> None:
    fieldnames = [
        "Ref_DOI_number",
        "Cell_architecture",
        "Cell_stack_sequence",
        "Perovskite_composition_short_form",
        "Perovskite_composition_long_form",
        "Perovskite_band_gap",
        "Perovskite_deposition_procedure",
        "Perovskite_deposition_thermal_annealing_temperature",
        "Perovskite_deposition_thermal_annealing_time",
        "ETL_stack_sequence",
        "HTL_stack_sequence",
        "Backcontact_stack_sequence",
        "JV_light_intensity",
        "JV_default_PCE",
        "JV_default_Voc",
        "JV_default_Jsc",
        "JV_default_FF",
    ]
    base_row = {
        "Cell_architecture": "nip",
        "Cell_stack_sequence": "SLG | FTO | TiO2 | Perovskite | Spiro-MeOTAD | Au",
        "Perovskite_composition_short_form": "MAPbI",
        "Perovskite_composition_long_form": "MAPbI3",
        "Perovskite_band_gap": "1.55",
        "Perovskite_deposition_procedure": "Spin-coating",
        "Perovskite_deposition_thermal_annealing_temperature": "100",
        "Perovskite_deposition_thermal_annealing_time": "10",
        "ETL_stack_sequence": "TiO2",
        "HTL_stack_sequence": "Spiro-MeOTAD",
        "Backcontact_stack_sequence": "Au",
        "JV_light_intensity": "100",
        "JV_default_PCE": "12",
        "JV_default_Voc": "1.0",
        "JV_default_Jsc": "20",
        "JV_default_FF": "0.6",
    }

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("train", "validation", "test"):
            row = dict(base_row)
            row["Ref_DOI_number"] = split_keys[split]
            writer.writerow(row)
