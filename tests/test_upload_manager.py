"""
tests/test_upload_manager.py — CSV Training Upload Validation (AI-022, FR-30) Unit Tests
=========================================================================================
Sheet 10_AI_MODULE row AI-022 asks that the SRS 5.6.7 24-month training window
exclude old rows *visibly*: a mixed-age CSV must come back reporting how many rows
were dropped and why, and must train only on what survived — the same
"don't hide it" principle applied to every other data-integrity fix in this project.

The cap lives in ``data/upload_manager.py`` (``_MAX_MONTHS_BACK = 24``), which is
what POST /api/ai/train calls before it queues a training job. The row's assertion
is written against the HTTP response (``skippedRows.tooOld``), but the counts are
produced by ``UploadManager.save_upload()`` and passed straight through as the
job's ``uploadSummary``, so both layers are checked here: the manager directly (so a
failure names the real cause) and then the same CSV through the endpoint.

Note on ``badDates`` vs ``tooOld``: unparseable dates are removed *before* the age
cutoff is applied, so the two counters cannot double-count the same row. The
17-row fixture below is built so that distinction is observable — 10 recent,
5 well outside the window, 2 with a date the parser cannot read.

Run:
    cd slt-ai-module
    venv/Scripts/python.exe -m pytest tests/test_upload_manager.py -v
"""

import io
import os
import sys
import json
from datetime import date

import pytest
from werkzeug.datastructures import FileStorage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.upload_manager import UploadManager, _MAX_MONTHS_BACK


RECENT_ROWS   = 10
TOO_OLD_ROWS  = 5
BAD_DATE_ROWS = 2

# Four in-bounds Sri Lanka coordinates, cycled so no row is dropped for GPS.
_COORDS = [(6.9271, 79.8612), (7.2906, 80.6337), (6.0535, 80.2210), (9.6615, 80.0255)]


def _mixed_age_csv() -> bytes:
    """
    17 rows: 10 inside the 24-month window, 5 comfortably outside it
    (30 months back — the cap is 24), and 2 whose date cannot be parsed at all.
    """
    today = date.today()
    lines = ['date,latitude,longitude,category']

    def row(day_str, i):
        lat, lng = _COORDS[i % len(_COORDS)]
        return f'{day_str},{lat},{lng},BROADBAND'

    for i in range(RECENT_ROWS):
        d = date(today.year, today.month, 1)
        # Spread across recent days without leaving the current month.
        lines.append(row(f'{d.year:04d}-{d.month:02d}-{(i % 28) + 1:02d}', i))

    for i in range(TOO_OLD_ROWS):
        year  = today.year - 3            # ~36 months back, well past the 24-month cap
        lines.append(row(f'{year:04d}-0{(i % 9) + 1}-15', i))

    for i in range(BAD_DATE_ROWS):
        lines.append(row('not-a-date', i))

    return ('\n'.join(lines) + '\n').encode()


def _as_upload(csv_bytes: bytes, filename: str = 'faults.csv') -> FileStorage:
    return FileStorage(stream=io.BytesIO(csv_bytes), filename=filename)


def test_24_month_cap_visible_exclusion():
    """AI-022 — excluded rows are counted and reported, not silently dropped."""
    summary, df = UploadManager().save_upload(_as_upload(_mixed_age_csv()))

    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    skipped = summary.get('skippedRows', {})

    # ── Step 2: the two exclusion reasons are reported separately ─────────────
    check(
        skipped.get('tooOld') == TOO_OLD_ROWS,
        f"skippedRows.tooOld == {skipped.get('tooOld')!r}, expected {TOO_OLD_ROWS} — "
        "rows outside the 24-month window must be counted, not silently discarded",
    )
    check(
        skipped.get('badDates') == BAD_DATE_ROWS,
        f"skippedRows.badDates == {skipped.get('badDates')!r}, expected {BAD_DATE_ROWS}",
    )

    # ── Step 3: the window is stated, so the UI can explain the exclusion ─────
    check(
        summary.get('maxTrainingWindowMonths') == _MAX_MONTHS_BACK == 24,
        f"maxTrainingWindowMonths == {summary.get('maxTrainingWindowMonths')!r}, "
        "expected 24 — without it the UI cannot tell the admin why rows vanished",
    )

    # ── Step 4: only the in-window rows are handed to training ───────────────
    check(
        summary.get('rowCount') == RECENT_ROWS,
        f"rowCount == {summary.get('rowCount')!r}, expected {RECENT_ROWS}",
    )
    check(
        len(df) == RECENT_ROWS,
        f"The DataFrame passed to training has {len(df)} rows, expected {RECENT_ROWS}",
    )

    cutoff_year = date.today().year - 2
    check(
        len(df) == 0 or df['date'].min().year >= cutoff_year,
        f"An excluded row survived into the training set: earliest date is "
        f"{None if len(df) == 0 else df['date'].min().date()}",
    )

    assert not failures, "\n".join(f"  - {f}" for f in failures)


def test_24_month_cap_reported_through_the_train_endpoint(client):
    """
    The same counts must reach the admin over HTTP, since that is where the row's
    own assertion is written (``body('skippedRows.tooOld', equalTo(5))``).
    Uses conftest.py's session-scoped Flask client (DB mocked unavailable).
    """
    resp = client.post(
        '/api/ai/train',
        data={'file': (io.BytesIO(_mixed_age_csv()), 'faults.csv')},
        content_type='multipart/form-data',
    )
    assert resp.status_code == 202, resp.data[:400]

    summary = json.loads(resp.data.decode('utf-8'))['data']['uploadSummary']
    assert summary['skippedRows']['tooOld']   == TOO_OLD_ROWS
    assert summary['skippedRows']['badDates'] == BAD_DATE_ROWS
    assert summary['maxTrainingWindowMonths'] == 24
    assert summary['rowCount'] == RECENT_ROWS


# ═════════════════════════════════════════════════════════════════════════════
# H3 — EXCHANGEAREA / OPMCCODE columns (the real WFMS export shape: no GPS
# at all, only short exchange-area codes; confirmed against a real sample).
# Additive: must not change behavior for CSVs that DO have latitude/longitude.
# ═════════════════════════════════════════════════════════════════════════════

# Matches the real WFMS sample's observed values.
_AREA_CODES = ['DGD', 'AD', 'KY', 'CEN', 'MHG', 'KG']
_OPMC_CODES = ['KTOP', 'ADOP', 'KYOP', 'MDOP', 'HOOP', 'KUOP']


def _exchange_area_only_csv(n_rows: int = 12) -> bytes:
    """
    The real WFMS export shape — EXCHANGEAREA and OPMCCODE columns (uppercase,
    matching the real header exactly), no latitude/longitude at all.
    """
    today = date.today()
    lines = ['DATE,EXCHANGEAREA,OPMCCODE,CATEGORY']
    for i in range(n_rows):
        d = date(today.year, today.month, (i % 27) + 1)
        area = _AREA_CODES[i % len(_AREA_CODES)]
        opmc = _OPMC_CODES[i % len(_OPMC_CODES)]
        lines.append(f'{d.isoformat()},{area},{opmc},BROADBAND')
    return ('\n'.join(lines) + '\n').encode()


def test_exchangearea_and_opmccode_headers_recognised_case_insensitively():
    """
    The real export's headers are uppercase (EXCHANGEAREA, OPMCCODE) — the
    alias system already lower-cases before matching (_normalise_columns),
    so this must work with no per-case special-casing.
    """
    summary, df = UploadManager().save_upload(_as_upload(_exchange_area_only_csv()))
    assert 'exchange_area' in df.columns, f"Columns after normalisation: {list(df.columns)}"
    assert 'opmc_code' in df.columns, f"Columns after normalisation: {list(df.columns)}"


def test_exchange_area_only_csv_is_usable_for_clustering():
    """
    A CSV with EXCHANGEAREA but no GPS must still be usable for clustering
    (via the categorical fallback), not silently marked unusable the way a
    GPS-less CSV was before this change.
    """
    summary, _ = UploadManager().save_upload(_as_upload(_exchange_area_only_csv()))
    assert summary['hasGpsColumns'] is False
    assert summary['hasExchangeAreaColumn'] is True
    assert summary['hasOpmcCodeColumn'] is True
    assert summary['usableForClustering'] is True, (
        "A CSV with EXCHANGEAREA must be usableForClustering via the categorical "
        "fallback, even with zero GPS columns"
    )
    assert summary['clusteringMethod'] == 'categorical'


def test_gps_still_wins_when_both_columns_present():
    """GPS is strictly more informative — a CSV with BOTH must prefer K-Means."""
    today = date.today()
    lines = ['date,latitude,longitude,exchangearea,category']
    for i in range(12):
        d = date(today.year, today.month, (i % 27) + 1)
        lat, lng = _COORDS[i % len(_COORDS)]
        area = _AREA_CODES[i % len(_AREA_CODES)]
        lines.append(f'{d.isoformat()},{lat},{lng},{area},BROADBAND')
    csv_bytes = ('\n'.join(lines) + '\n').encode()

    summary, df = UploadManager().save_upload(_as_upload(csv_bytes))
    assert summary['hasGpsColumns'] is True
    assert summary['hasExchangeAreaColumn'] is True
    assert summary['clusteringMethod'] == 'kmeans', (
        "GPS must take priority over EXCHANGEAREA when a CSV has both"
    )


def test_to_exchange_area_groups_shapes_the_data():
    summary, df = UploadManager().save_upload(_as_upload(_exchange_area_only_csv(n_rows=12)))
    area_df = UploadManager().to_exchange_area_groups(df)
    assert area_df is not None
    for col in ['exchange_area', 'opmc_code', 'category', 'created_at']:
        assert col in area_df.columns, f"Missing column: {col}"
    assert len(area_df) == 12
    assert set(area_df['exchange_area'].unique()) == set(_AREA_CODES)


def test_to_exchange_area_groups_returns_none_without_the_column():
    """Regression: a plain GPS-only CSV (today's existing shape) is untouched."""
    summary, df = UploadManager().save_upload(_as_upload(_mixed_age_csv()))
    assert UploadManager().to_exchange_area_groups(df) is None


def test_to_gps_points_unaffected_by_exchange_area_columns_being_absent():
    """Regression: existing GPS shaping path is unchanged by this feature."""
    summary, df = UploadManager().save_upload(_as_upload(_mixed_age_csv()))
    gps_df = UploadManager().to_gps_points(df)
    assert gps_df is not None
    assert len(gps_df) == RECENT_ROWS
    for col in ['latitude', 'longitude', 'category', 'created_at']:
        assert col in gps_df.columns


def test_opmc_code_does_not_match_any_real_opmc_in_this_database():
    """
    Documents the bonus-finding check, not a live DB assertion (upload_manager
    does not query the database — see its module docstring for why: this
    database's real Opmc.code values, confirmed via a direct query, are
    ABC-01/TES-10/NEG-18; the real WFMS export's OPMCCODE values are
    KTOP/ADOP/KYOP/MDOP/HOOP/KUOP. Disjoint sets, confirmed by inspection —
    this test pins that opmc_code is carried through as a plain string
    column, not silently coerced or validated against anything.
    """
    summary, df = UploadManager().save_upload(_as_upload(_exchange_area_only_csv()))
    real_opmc_codes = {'ABC-01', 'TES-10', 'NEG-18'}
    uploaded_codes = set(df['opmc_code'].unique())
    assert uploaded_codes.isdisjoint(real_opmc_codes), (
        "This test's fixture codes were expected to differ from the real seeded "
        f"Opmc.code values; got overlap {uploaded_codes & real_opmc_codes}"
    )
    assert uploaded_codes == set(_OPMC_CODES)


def test_all_rows_too_old_is_rejected_outright():
    """
    The boundary the row implies but does not spell out: if the cap leaves
    nothing behind, the upload must fail with a message naming the window
    rather than queueing a job that trains on zero rows.
    """
    year  = date.today().year - 3
    lines = ['date,latitude,longitude,category']
    for i in range(6):
        lat, lng = _COORDS[i % len(_COORDS)]
        lines.append(f'{year:04d}-0{(i % 9) + 1}-15,{lat},{lng},FIBER')
    csv_bytes = ('\n'.join(lines) + '\n').encode()

    with pytest.raises(ValueError) as exc:
        UploadManager().save_upload(_as_upload(csv_bytes))
    assert '24-month' in str(exc.value) or '2-year' in str(exc.value), \
        f"Rejection message does not name the window: {exc.value}"
