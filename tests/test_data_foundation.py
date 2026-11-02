import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import build_target, clean, parse_issue_date, target_summary
from src.data.ingestion import load_csv, load_parquet, save_parquet
from src.data.leakage import flag_suspicious, split_features


def _raw():
    return pd.DataFrame({
        "loan_status": ["Fully Paid", "Charged Off", "Current", "Default", "Late (31-120 days)",
                        "Fully Paid", "Fully Paid"],
        "issue_d": ["Jan-2015", "Feb-2015", "Mar-2016", "Apr-2016", "May-2016", "Jan-2015", "bad"],
        "loan_amnt": [1000, 2000, 3000, 4000, 5000, 1000, 6000],
    })


def test_target_excludes_immature_statuses():
    out = build_target(_raw())
    assert set(out["target"]) == {0, 1}
    assert len(out) == 5  # Current и Late исключены
    assert out.loc[out["loan_status"] == "Default", "target"].iloc[0] == 1


def test_target_missing_column():
    with pytest.raises(KeyError):
        build_target(pd.DataFrame({"a": [1]}))


def test_parse_issue_date_invalid_to_nat():
    out = parse_issue_date(_raw())
    assert out["issue_d"].isna().sum() == 1


def test_clean_removes_duplicates_and_bad_dates():
    out = clean(_raw())
    # дубликат (Fully Paid, Jan-2015, 1000) и строка с некорректной датой удалены
    assert len(out) == 3
    summary = target_summary(out)
    assert summary.loc[2015, "n"] == 2


def test_split_features():
    res = split_features(["loan_amnt", "total_pymnt", "mystery"])
    assert res == {"allowed": ["loan_amnt"], "leakage": ["total_pymnt"], "unknown": ["mystery"]}


def test_flag_suspicious_detects_leaky_feature():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    df = pd.DataFrame({"target": y, "leaky": y + rng.normal(0, 0.05, 500),
                       "noise": rng.normal(size=500)})
    res = flag_suspicious(df, ["leaky", "noise"]).set_index("feature")
    assert res.loc["leaky", "suspicious"]
    assert not res.loc["noise", "suspicious"]


def test_flag_suspicious_needs_both_classes():
    df = pd.DataFrame({"target": [0] * 100, "x": np.arange(100)})
    with pytest.raises(ValueError):
        flag_suspicious(df, ["x"])


def test_parquet_roundtrip(tmp_path):
    csv = tmp_path / "a.csv"
    _raw().to_csv(csv, index=False)
    raw = load_csv(csv, as_str=True)
    assert raw["loan_amnt"].iloc[0] == "1000"  # значения не преобразованы
    path = save_parquet(raw, tmp_path / "sub" / "a.parquet")
    pd.testing.assert_frame_equal(load_parquet(path), raw)


def test_load_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv")


def test_build_target_does_not_mutate_input():
    raw = _raw()
    before = raw.copy()
    build_target(raw)
    pd.testing.assert_frame_equal(raw, before)  # исходный DataFrame без изменений


def test_clean_on_wide_frame_keeps_all_columns():
    raw = _raw()
    extra = pd.DataFrame({f"c{i}": i for i in range(150)}, index=raw.index)
    raw = pd.concat([raw, extra], axis=1)
    out = clean(raw)
    assert out.shape[1] == raw.shape[1] + 1  # + target
    assert len(out) == 3


def test_prepare_reads_only_working_columns_and_converts_types(tmp_path):
    from src.data.prepare import prepare, working_columns

    raw = _raw().astype(str)
    raw["int_rate"] = ["10.5", "11.0", "12.0", "13.0", "14.0", "10.5", "bad"]
    raw["total_pymnt"] = "1"  # признак утечки: не должен попасть в результат
    raw["id"] = [str(i) for i in range(len(raw))]
    raw["term"] = " 36 months"
    path = save_parquet(raw, tmp_path / "raw.parquet")
    out = prepare(path, tmp_path / "processed" / "clean.parquet")
    assert "total_pymnt" not in out.columns
    assert set(out.columns) <= set(working_columns()) | {"target"}
    assert str(out["int_rate"].dtype) == "float32"
    assert (tmp_path / "processed" / "clean.parquet").exists()

