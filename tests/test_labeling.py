import json

import pytest

from hm import labeling


def _job(root, name, segs, sem):
    d = root / name
    d.mkdir()
    (d / "storyboard.json").write_text(json.dumps({"segments": segs}), encoding="utf-8")
    (d / "semantics.json").write_text(json.dumps(sem), encoding="utf-8")
    (d / "final_cropped.mp4").write_bytes(b"x")


def _seg(i, sel=True):
    return {"segment_id": "seg_00%d" % i, "source_text": "文%d" % i, "text_gen_final": "p%d" % i,
            "selected": "cand_01" if sel else None}


@pytest.fixture
def root(tmp_path):
    _job(tmp_path, "j1", [_seg(1), _seg(2), _seg(3), _seg(4, sel=False)],
         [{"segment": "001", "status": "通过"}, {"segment": "002", "status": "通过"},
          {"segment": "003", "status": "可疑"}])
    return tmp_path


def test_export_is_blind_and_skips_unselected(root, tmp_path):
    rows = labeling.export_rows(root)
    assert [r["seg"] for r in rows] == ["001", "002", "003"]
    out = tmp_path / "l.csv"
    labeling.write_csv(out, rows)
    text = out.read_text(encoding="utf-8-sig")
    assert "通过" not in text and "可疑" not in text      # 判据结论不能泄露进标注表
    assert "final_cropped.mp4" in text


def test_score_leak_and_false_alarm(root, tmp_path):
    out = tmp_path / "l.csv"
    rows = labeling.export_rows(root)
    for r, v in zip(rows, ["是", "n", "yes"]):
        r[labeling.LABEL_COL] = v
    labeling.write_csv(out, rows)
    r = labeling.score(root, labeling.read_labels(out))
    assert r["n"] == 3
    assert r["leak_n"] == (1, 2) and r["leak_rate"] == 0.5          # 002 通过但人判否
    assert r["false_alarm_n"] == (1, 1)                              # 003 可疑但人判是
    assert abs(r["correct_rate"] - 2 / 3) < 1e-9
    assert "语义漏放" in labeling.format_report(r)


def test_blank_rows_ignored_and_bad_value_rejected(root, tmp_path):
    rows = labeling.export_rows(root)
    rows[0][labeling.LABEL_COL] = "也许"
    out = tmp_path / "l.csv"
    labeling.write_csv(out, rows)
    with pytest.raises(ValueError):
        labeling.read_labels(out)
    rows[0][labeling.LABEL_COL] = ""
    labeling.write_csv(out, rows)
    assert labeling.read_labels(out) == {}


def test_label_without_semantics_entry_is_reported(root, tmp_path):
    rows = labeling.export_rows(root)
    rows[0][labeling.LABEL_COL] = "是"
    out = tmp_path / "l.csv"
    labeling.write_csv(out, rows)
    labels = labeling.read_labels(out)
    labels[("ghost", "009")] = {"label": "否", "action": "", "source_text": ""}
    assert labeling.score(root, labels)["missing"] == [("ghost", "009")]
