"""
tests/test_formatters.py — LKR Currency Formatter (PAY-016, FR-10) Unit Tests
=============================================================================
Sheet 04_PAYMENT_FLOW row PAY-016 asks that the LKR formatter render an ordinary amount, zero,
a millions-scale amount with a fractional part, and ``None`` — every case with the ``LKR`` prefix,
comma thousands separators and exactly two decimal places.

The formatter under test is ``utils/formatters.fmt_lkr``, which is where the row's Tool column
(PyTest) and Automation Mapping (``test_formatters.py::test_fmt_lkr_all_cases``) point. Note that
this is one of THREE independent LKR formatters in the system — the others are
``SLTMobileApp/src/utils/formatters.ts::formatCurrency`` (``toLocaleString('en-US', ...)``) and
``frontend-admin/src/pages/Payments/PaymentsPage.js::fmtLKR`` (``toLocaleString('en-LK', ...)``,
and the only one of the three that renders ``None``/``null`` as an em dash rather than
``LKR 0.00``). Only the Python one is in this row's scope.

All four of the row's cases are evaluated and reported together via subtests rather than aborting
on the first failure, so a single run names every case that is wrong. A handful of adjacent
behaviours the row implies but does not spell out (negative amounts, the ``symbol``/``decimals``
switches, non-numeric input) are covered in the companion tests below, since the row's headline
claim is "correct for all cases".

Run:
    cd slt-ai-module
    venv/Scripts/python.exe -m pytest tests/test_formatters.py -v
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.formatters import fmt_lkr


def test_fmt_lkr_all_cases():
    """PAY-016 — the four cases the row specifies, each reported independently."""
    cases = [
        # (input,      expected,             why it is in the row)
        (3800.00,      'LKR 3,800.00',       'ordinary amount: prefix + thousands separator + 2dp'),
        (0,            'LKR 0.00',           'zero must still render 2dp, not "LKR 0"'),
        (1000000.50,   'LKR 1,000,000.50',   'millions scale: separator every 3 digits, fraction kept'),
        (None,         'LKR 0.00',           'a missing amount must degrade to zero, never crash or "None"'),
    ]

    failures = []
    for value, expected, why in cases:
        try:
            actual = fmt_lkr(value)
        except Exception as exc:                      # noqa: BLE001 — report, do not abort
            failures.append(f'fmt_lkr({value!r}) raised {exc!r} — {why}')
            continue
        if actual != expected:
            failures.append(
                f'fmt_lkr({value!r}) == {actual!r}, expected {expected!r} — {why}'
            )

    assert not failures, (
        'LKR formatting is wrong for '
        f'{len(failures)} of {len(cases)} cases:\n  ' + '\n  '.join(failures)
    )


def test_fmt_lkr_negative_amount_keeps_sign_outside_the_prefix():
    """A credit/refund must read '-LKR 250.00', not 'LKR -250.00' — the sign leads the string."""
    assert fmt_lkr(-250) == '-LKR 250.00'


def test_fmt_lkr_symbol_and_decimals_switches():
    """The two optional switches must not disturb the separator or the rounding."""
    assert fmt_lkr(3800.00, symbol=False) == '3,800.00'
    assert fmt_lkr(3800.456, decimals=0) == 'LKR 3,800'
    # Rounding is Python's format(), i.e. round-half-even over the binary float — 3800.455 is
    # really 3800.45499..., so it renders .45. Pinned deliberately: money rendered from a float
    # is not exact, and a caller needing exact half-up must pass a Decimal upstream.
    assert fmt_lkr(3800.455, decimals=2) == 'LKR 3,800.45'


def test_fmt_lkr_non_numeric_input_degrades_to_zero():
    """Garbage in must not propagate an exception into a JSON response."""
    assert fmt_lkr('not-a-number') == 'LKR 0.00'
    assert fmt_lkr('not-a-number', symbol=False) == '0.00'
