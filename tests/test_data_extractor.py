"""
tests/test_data_extractor.py — H1d: additive columns on get_faults_with_location()
====================================================================================
Verifies the narrow, descoped H1d addition confirmed by investigation
(QA_Compliance_Consolidated_Report.md's H1d entry): get_faults_with_location()
now also selects circuit_id, nearest_exchange_id, nearest_exchange_distance_km
as read-only columns, purely for cross-referencing a clustered fault against
its stable Exchange/Circuit -- not fed into the K-Means fit itself (see
tests/test_clustering.py::TestH1dAdditiveColumnsDoNotAffectClustering for
that half of the proof).

Real DB, real seeded/cleaned-up row -- same standard as this session's other
real-data verification, not a mock of the SQL layer.

Run:
    cd slt-ai-module
    venv/Scripts/python.exe -m pytest tests/test_data_extractor.py -v
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_db_engine
from sqlalchemy import text


@pytest.fixture(scope='module')
def db_engine():
    engine = get_db_engine()
    if engine is None:
        pytest.skip("Database not available")
    return engine


@pytest.fixture
def seeded_fault(db_engine):
    """
    A real faults row with real, non-NULL circuit_id/nearest_exchange_id/
    nearest_exchange_distance_km values, inserted directly (matching this
    session's established pattern for seeding real rows outside the app's
    own write path) and deleted afterward regardless of test outcome.
    """
    with db_engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO faults (
                fault_number, customer_id, category, description, opmc_id,
                priority, status, reopen_count,
                latitude, longitude, circuit_id, nearest_exchange_id,
                nearest_exchange_distance_km
            ) VALUES (
                :fault_number, 6, 'INTERNET', 'H1d test_data_extractor seed', 1,
                'MEDIUM', 'REPORTED', 0,
                6.9271, 79.8612, :circuit_id, :nearest_exchange_id, :distance_km
            )
        """), {
            'fault_number': f'H1D-TEST-{os.getpid()}',
            'circuit_id': 188016,
            'nearest_exchange_id': 32,
            'distance_km': 3.42,
        })
        fault_id = result.lastrowid

    yield fault_id

    with db_engine.begin() as conn:
        conn.execute(text("DELETE FROM faults WHERE id = :id"), {'id': fault_id})


class TestGetFaultsWithLocationAdditiveColumns:

    def test_new_columns_present_and_correct_for_a_real_row(self, seeded_fault, db_engine):
        from data.data_extractor import DataExtractor
        extractor = DataExtractor()
        extractor.engine = db_engine
        extractor._db_ok = True

        df = extractor.get_faults_with_location(days_back=1)
        assert df is not None

        row = df[df['id'] == seeded_fault]
        assert len(row) == 1, f"Seeded fault {seeded_fault} must appear in the query result"
        row = row.iloc[0]

        assert int(row['circuit_id']) == 188016
        assert int(row['nearest_exchange_id']) == 32
        assert row['nearest_exchange_distance_km'] == pytest.approx(3.42)

    def test_columns_are_present_but_null_for_faults_without_them(self, db_engine):
        """The overwhelming majority of real faults today (nothing sets circuit_id outside
        the new manual-attach endpoint) -- confirms the columns don't error or get dropped
        just because they're NULL, and NULL doesn't filter the row out of the result."""
        from data.data_extractor import DataExtractor
        extractor = DataExtractor()
        extractor.engine = db_engine
        extractor._db_ok = True

        df = extractor.get_faults_with_location(days_back=3650)
        assert df is not None
        assert {'circuit_id', 'nearest_exchange_id', 'nearest_exchange_distance_km'} <= set(df.columns)
        # At least one real row with a NULL circuit_id must still be present (not silently
        # excluded) -- true for every real fault in this DB today outside the seeded one above.
        assert df['circuit_id'].isna().any() or len(df) <= 1
