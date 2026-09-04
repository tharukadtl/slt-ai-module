"""
tests/test_fault_validators.py — GPS Coordinate Validation (FAULT-004, FR-4) Unit Tests
========================================================================================
Sheet 02_FAULT_TRACKING row FAULT-004 asks for `validate_coords()` to accept a point
inside Sri Lanka, reject a point outside it, treat absent coordinates as optional, and
reject a latitude below the southern bound.

The validator lives in this module, not in the Spring backend: `utils/validators.py`
defines `validate_coords(lat_raw, lng_raw) -> (lat, lng, error)`, layered over
`validate_latitude` / `validate_longitude`, which clamp to Config.SL_LAT_MIN/MAX
(5.9-9.9) and Config.SL_LNG_MIN/MAX (79.5-81.9). The fieldops backend has no equivalent
geographic check anywhere (see FAULT-018 / LocationServiceTest), so this is the only
place the Sri Lanka bounds are enforced in the system.

All four steps are evaluated and reported together rather than aborting on the first
failure, because step 3 is expected to fail: `validate_coords` calls both underlying
validators with `required=True`, so `(None, None)` returns an "is required" error rather
than being accepted as optional.

Run:
    cd slt-ai-module
    venv/Scripts/python.exe -m pytest tests/test_fault_validators.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from utils.validators import validate_coords


# ── Test data from the row ────────────────────────────────────────────────────
COLOMBO = (6.9271, 79.8612)      # inside Sri Lanka
LONDON = (51.5, -0.12)           # far outside
BELOW_MIN_LAT = (5.8, 80.0)      # 5.8 < SL_LAT_MIN (5.9)


def test_validate_coords_sl_bounds():
    """FAULT-004 — Sri Lanka GPS bounds, with absent coordinates treated as optional."""
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    # ── Step 1: a Colombo coordinate is valid ─────────────────────────────────
    lat, lng, err = validate_coords(*COLOMBO)
    check(
        err is None,
        f"Step 1: {COLOMBO} is inside Sri Lanka and must validate, got error: {err!r}",
    )
    check(
        (lat, lng) == COLOMBO,
        f"Step 1: valid coordinates must be returned unchanged, got ({lat}, {lng})",
    )

    # ── Step 2: London is outside Sri Lanka and must be rejected ──────────────
    lat, lng, err = validate_coords(*LONDON)
    check(
        err is not None,
        f"Step 2: {LONDON} (London) is outside Sri Lanka and must be rejected, got error: {err!r}",
    )
    check(
        err is not None and "Sri Lanka" in err,
        f"Step 2: the rejection must say the point is outside Sri Lanka, got: {err!r}",
    )
    check(
        (lat, lng) == (None, None),
        f"Step 2: a rejected coordinate must not leak a value, got ({lat}, {lng})",
    )

    # ── Step 3: absent coordinates are optional ───────────────────────────────
    lat, lng, err = validate_coords(None, None)
    check(
        err is None,
        "Step 3: (None, None) must be accepted as optional. validate_coords calls "
        "validate_latitude/validate_longitude with required=True, so absent coordinates "
        "are rejected as missing rather than allowed. "
        f"Got error: {err!r}",
    )

    # ── Step 4: a latitude below the southern bound is rejected ───────────────
    lat, lng, err = validate_coords(*BELOW_MIN_LAT)
    check(
        err is not None,
        f"Step 4: latitude {BELOW_MIN_LAT[0]} is below SL_LAT_MIN "
        f"({Config.SL_LAT_MIN}) and must be rejected, got error: {err!r}",
    )
    check(
        err is not None and str(Config.SL_LAT_MIN) in err,
        f"Step 4: the rejection should name the minimum latitude "
        f"({Config.SL_LAT_MIN}), got: {err!r}",
    )

    assert not failures, "\n".join(f"  - {f}" for f in failures)
