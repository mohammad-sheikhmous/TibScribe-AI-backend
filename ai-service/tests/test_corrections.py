"""Doctor corrections, the review queue and the export loop (P7-01/04/07).

The properties under test are the ones that make the loop trustworthy: a correction
never destroys what the model said, the queue puts the most informative sentence first,
and every exported row knows where it came from.
"""
import json

import pytest

from app.db import repo, session_scope


def _upload(client, wav_bytes, **data):
    return client.post(
        "/jobs", files={"file": ("v.wav", wav_bytes, "audio/wav")}, data=data or None
    ).json()


def _first_item(client, job_id):
    report = client.get(f"/jobs/{job_id}/report").json()
    return next(it for section in report["soap"].values() for it in section["items"])


# --- correcting an item ---------------------------------------------------------------

def test_label_correction_updates_the_report(client, wav_bytes):
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    assert item["label"] == "symptom"

    resp = client.patch(f"/jobs/{job_id}/items/{item['item_id']}",
                        json={"label": "diagnosis", "actor": "د. سارة"})
    assert resp.status_code == 200

    corrected = _first_item(client, job_id)
    assert corrected["label"] == "diagnosis"
    assert corrected["soap_section"] == "assessment"      # section follows the label
    assert corrected["label_ar"] == "التشخيص"


def test_the_original_value_is_never_destroyed(client, wav_bytes):
    """A correction is an append: the (predicted -> corrected) pair IS the training signal."""
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    client.patch(f"/jobs/{job_id}/items/{item['item_id']}", json={"label": "vital"})

    history = client.get(f"/jobs/{job_id}/corrections").json()
    assert len(history) == 1
    assert history[0]["old_value"] == "symptom"          # what the model said
    assert history[0]["new_value"] == "vital"            # what the doctor said
    assert history[0]["field"] == "label"


def test_correcting_clears_the_review_flag(client, wav_bytes):
    """A human decision is certain — it must stop being queued for the review it had."""
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    client.patch(f"/jobs/{job_id}/items/{item['item_id']}", json={"label": "vital"})

    corrected = _first_item(client, job_id)
    assert corrected["is_low_confidence"] is False
    assert corrected["low_confidence_reasons"] == []


def test_text_and_speaker_can_also_be_corrected(client, wav_bytes):
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    resp = client.patch(f"/jobs/{job_id}/items/{item['item_id']}",
                        json={"text": "صداع شديد منذ يومين", "speaker": "patient"})
    assert resp.status_code == 200
    assert {c["field"] for c in resp.json()} == {"text", "speaker"}


def test_several_fields_produce_several_records(client, wav_bytes):
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    resp = client.patch(f"/jobs/{job_id}/items/{item['item_id']}",
                        json={"label": "vital", "text": "الضغط 130 على 80"})
    assert len(resp.json()) == 2


def test_a_no_op_correction_is_rejected(client, wav_bytes):
    """Re-submitting the same value must not pollute the training data with a fake edit."""
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    resp = client.patch(f"/jobs/{job_id}/items/{item['item_id']}",
                        json={"label": item["label"]})
    assert resp.status_code == 409


def test_unknown_label_is_rejected(client, wav_bytes):
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    resp = client.patch(f"/jobs/{job_id}/items/{item['item_id']}",
                        json={"label": "not_a_real_label"})
    assert resp.status_code == 422


def test_empty_correction_is_rejected(client, wav_bytes):
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    assert client.patch(f"/jobs/{job_id}/items/{item['item_id']}", json={}).status_code == 422


def test_unknown_item_is_404(client, wav_bytes):
    job_id = _upload(client, wav_bytes)["job_id"]
    assert client.patch(f"/jobs/{job_id}/items/nope", json={"label": "vital"}).status_code == 404


# --- the review queue (P7-04) -----------------------------------------------------------

def test_review_queue_orders_by_uncertainty_not_document_order(temp_env):
    from tests.conftest import build_app
    from fastapi.testclient import TestClient

    class Pipeline:
        """Third sentence is the least certain — it must come first in the queue."""

        def process(self, audio_path, job_id, filename=None, patient_info=None):
            from app.core.report.builder import build_report
            from app.core.report.schema import ClassifiedSegment

            segs = [
                ClassifiedSegment(order_index=i, text=f"جملة {i}", start_sec=i,
                                  end_sec=i + 1, source_segment_index=i,
                                  label="symptom", confidence=conf,
                                  review_priority=priority)
                for i, (conf, priority) in enumerate([(0.95, 0.1), (0.9, 0.2), (0.4, 0.9)])
            ]
            return build_report(job_id, segs)

    app = build_app(pipeline=Pipeline())
    with TestClient(app) as client:
        job_id = client.post("/jobs", files={"file": ("v.wav", b"RIFFxx", "audio/wav")}
                             ).json()["job_id"]
        queue = client.get(f"/jobs/{job_id}/review-queue", params={"limit": 3}).json()
        assert [row["order_index"] for row in queue] == [2, 1, 0]
        assert queue[0]["review_priority"] == 0.9
    app.state.executor.shutdown(wait=False)


def test_corrected_items_leave_the_queue(client, wav_bytes):
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    assert client.get(f"/jobs/{job_id}/review-queue").json()

    client.patch(f"/jobs/{job_id}/items/{item['item_id']}", json={"label": "vital"})
    remaining = client.get(f"/jobs/{job_id}/review-queue").json()
    assert item["item_id"] not in {row["item_id"] for row in remaining}


def test_global_queue_spans_reports(client, wav_bytes):
    _upload(client, wav_bytes)
    _upload(client, wav_bytes + b"2")
    queue = client.get("/review-queue", params={"limit": 10}).json()
    assert len({row["job_id"] for row in queue}) == 2


# --- export (P7-07) ------------------------------------------------------------------------

def test_export_tags_every_row_with_its_source(client, wav_bytes, tmp_path):
    from scripts.export_training_data import export

    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    client.patch(f"/jobs/{job_id}/items/{item['item_id']}",
                 json={"label": "diagnosis", "actor": "د. أحمد"})

    out = tmp_path / "corrections.jsonl"
    stats = export(out, dry_run=False, include_confirmed=False, mark_exported=True)

    assert stats["rows"] == 1
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["source"] == "human"
    assert row["label"] == "diagnosis"
    assert row["was_predicted"] == "symptom"     # the pair, not just the answer
    assert row["actor"] == "د. أحمد"


def test_exported_corrections_are_not_exported_twice(client, wav_bytes, tmp_path):
    from scripts.export_training_data import export

    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    client.patch(f"/jobs/{job_id}/items/{item['item_id']}", json={"label": "vital"})

    first = export(tmp_path / "a.jsonl", dry_run=False, include_confirmed=False,
                   mark_exported=True)
    second = export(tmp_path / "b.jsonl", dry_run=False, include_confirmed=False,
                    mark_exported=True)
    assert first["rows"] == 1 and second["rows"] == 0


def test_dry_run_writes_nothing_and_marks_nothing(client, wav_bytes, tmp_path):
    from scripts.export_training_data import export

    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    client.patch(f"/jobs/{job_id}/items/{item['item_id']}", json={"label": "vital"})

    out = tmp_path / "dry.jsonl"
    export(out, dry_run=True, include_confirmed=False, mark_exported=True)
    assert not out.exists()
    assert export(tmp_path / "real.jsonl", dry_run=False, include_confirmed=False,
                  mark_exported=True)["rows"] == 1


# --- correction statistics (the non-blocked half of P7-03) ----------------------------

def test_correction_stats_expose_the_real_confusions(client, wav_bytes):
    job_id = _upload(client, wav_bytes)["job_id"]
    item = _first_item(client, job_id)
    client.patch(f"/jobs/{job_id}/items/{item['item_id']}", json={"label": "diagnosis"})

    with session_scope() as session:
        stats = repo.correction_stats(session)

    assert stats["items_corrected"] == 1
    assert 0 < stats["correction_rate"] <= 1
    assert stats["confusions"][0] == {"predicted": "symptom",
                                      "corrected_to": "diagnosis", "count": 1}
