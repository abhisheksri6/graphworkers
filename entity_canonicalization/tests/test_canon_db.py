"""DB-integration for the canonicalization engine (gated on DATABASE_URL):
  KG-AC-22 duplicates collapse to one canonical_id + idempotent re-run;
  KG-AC-25 batch-scoped (a folder outside the batch is untouched);
  KG-AC-38 cross-run reuse (a later batch matching a prior canonical reuses its id) + race-safe mint;
  KG-AC-39 single-instance over the WHOLE released batch (all folders in one call);
  KG-AC-40 atomic (a mid-batch failure rolls back to all-staged).
Uses unique folder ids and cleans up.
"""
import os
import uuid

import pytest

from ontologies import load_pack
from store import canonicalize_batch

_DSN = os.environ.get("DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not _DSN, reason="DATABASE_URL not set (DB-integration test)")
FIBO = load_pack("fibo_core")


def _norm(dsn):
    for p in ("postgresql+psycopg2://", "postgresql+asyncpg://", "postgresql+psycopg://"):
        if dsn.startswith(p):
            return "postgresql://" + dsn[len(p):]
    return dsn


class _DbShim:
    def __init__(self, conn):
        self._c = conn

    def connection(self):
        return type("_W", (), {"connection": self._c})()


@pytest.fixture
def conn():
    import psycopg2
    c = psycopg2.connect(_norm(_DSN))
    c.autocommit = False
    yield c
    c.close()


def _seed(cur, folder_id, mentions):
    """mentions: list of (surface, entity_type) OR (surface, entity_type, source_doc_id) OR
    (surface, entity_type, source_doc_id, attributes)."""
    import json as _json
    for i, item in enumerate(mentions):
        surface, etype = item[0], item[1]
        doc_id = item[2] if len(item) > 2 else None
        attrs = item[3] if len(item) > 3 else []
        uid = f"{folder_id}:{i}"
        cur.execute(
            """INSERT INTO public.kg_entities
                   (folder_id, entity_uid, entity_type, surface_form, source_doc_id, attributes,
                    ontology_pack, ontology_version, stage)
               VALUES (%s,%s,%s,%s,%s,%s,'fibo_core','1.0','staged')
               ON CONFLICT (entity_uid) DO NOTHING""",
            (folder_id, uid, etype, surface, doc_id, _json.dumps(attrs)),
        )


def _canonical_row(cur, canonical_key):
    cur.execute(
        "SELECT canonical_name, aliases FROM public.kg_canonical_entities WHERE canonical_key = %s",
        (canonical_key,),
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else (None, None)


def _canonical_attributes(cur, canonical_key):
    cur.execute(
        "SELECT attributes FROM public.kg_canonical_entities WHERE canonical_key = %s",
        (canonical_key,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _rows(cur, folder_id):
    cur.execute("SELECT surface_form, entity_type, canonical_id, stage FROM public.kg_entities WHERE folder_id=%s ORDER BY entity_uid", (folder_id,))
    return cur.fetchall()


def _cleanup(conn, folder_ids, canonical_keys=()):
    with conn.cursor() as cur:
        for f in folder_ids:
            cur.execute("DELETE FROM public.kg_entities WHERE folder_id=%s", (f,))
        for k in canonical_keys:
            cur.execute("DELETE FROM public.kg_canonical_entities WHERE canonical_key=%s", (k,))
    conn.commit()


@pytest.mark.ac("KG-AC-22")
@pytest.mark.ac("KG-AC-39")
@pytest.mark.ac("KG-AC-25")
def test_batch_collapses_dupes_single_instance_and_scoped(conn):
    fa, fb, fc = (f"canon-{uuid.uuid4()}" for _ in range(3))
    try:
        with conn.cursor() as cur:
            _seed(cur, fa, [("Acme Corp", "Organization"), ("Acme Corporation", "InvestmentAdviser")])
            _seed(cur, fb, [("Acme Corp", "Organization")])   # same entity, different folder
            _seed(cur, fc, [("Globex Ltd", "Organization")])  # OUTSIDE the batch
        conn.commit()

        # single-instance over the WHOLE released batch (fa + fb in one call), NOT fc
        summary = canonicalize_batch(_DbShim(conn), [fa, fb], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            ra, rb, rc = _rows(cur, fa), _rows(cur, fb), _rows(cur, fc)
        # KG-AC-22: all three Acme mentions (2 in fa + 1 in fb) share ONE canonical_id
        acme_cids = {r[2] for r in ra} | {r[2] for r in rb}
        assert len(acme_cids) == 1 and None not in acme_cids
        assert summary["canonical_count"] == 1  # one real entity across the batch
        # KG-AC-23 reconciled type is on the canonical index (InvestmentAdviser beats Organization)
        assert all(r[3] == "canonicalized" for r in ra + rb)
        # KG-AC-25: the out-of-batch folder is untouched (still staged, no canonical_id)
        assert all(r[3] == "staged" and r[2] is None for r in rc)

        # KG-AC-22 idempotent: a re-run finds nothing staged -> no-op
        summary2 = canonicalize_batch(_DbShim(conn), [fa, fb], pack=FIBO)
        conn.commit()
        assert summary2 == {"canonical_count": 0, "merged_count": 0, "minted_count": 0}
    finally:
        _cleanup(conn, [fa, fb, fc], ["investmentadviser:acme", "organization:acme", "organization:globex"])


@pytest.mark.ac("KG-AC-38")
def test_cross_run_reuses_existing_canonical(conn):
    f1, f2 = f"canon-{uuid.uuid4()}", f"canon-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f1, [("Acme Corp", "Organization")])
        conn.commit()
        s1 = canonicalize_batch(_DbShim(conn), [f1], pack=FIBO)
        conn.commit()
        assert s1["minted_count"] == 1  # novel canonical minted
        with conn.cursor() as cur:
            cid1 = _rows(cur, f1)[0][2]

        # a LATER batch with the same entity in a new folder -> reuse the existing canonical (merged)
        with conn.cursor() as cur:
            _seed(cur, f2, [("Acme Corporation", "Organization")])
        conn.commit()
        s2 = canonicalize_batch(_DbShim(conn), [f2], pack=FIBO)
        conn.commit()
        assert s2["minted_count"] == 0 and s2["merged_count"] == 1
        with conn.cursor() as cur:
            cid2 = _rows(cur, f2)[0][2]
        assert cid2 == cid1  # the two become one graph node across runs
    finally:
        _cleanup(conn, [f1, f2], ["organization:acme"])


@pytest.mark.ac("KG-AC-76")
@pytest.mark.ac("KG-AC-77")
def test_canonical_name_and_aliases_written_on_mint(conn):
    fa = f"canon-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, fa, [("Acme", "Organization"), ("Acme Corporation", "Organization"),
                            ("Acme Corp", "Organization")])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [fa], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            name, aliases = _canonical_row(cur, "organization:acme")
        assert name == "Acme Corporation"  # longest complete surface
        assert set(aliases) == {"Acme", "Acme Corp"}
        assert "Acme Corporation" not in aliases
    finally:
        _cleanup(conn, [fa], ["organization:acme"])


@pytest.mark.ac("KG-AC-77")
def test_aliases_grow_across_a_cross_run_merge(conn):
    # KG-AC-77's stated purpose is "every string that resolved to this instance" -- a LATER batch
    # (KG-AC-38 cross-run reuse) introducing a new surface variant must be reflected in aliases,
    # not just the founding batch's cluster.
    f1, f2 = f"canon-{uuid.uuid4()}", f"canon-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f1, [("Acme Corp", "Organization")])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f1], pack=FIBO)
        conn.commit()
        with conn.cursor() as cur:
            name1, aliases1 = _canonical_row(cur, "organization:acme")
        assert name1 == "Acme Corp" and aliases1 == []

        with conn.cursor() as cur:
            _seed(cur, f2, [("Acme Corporation", "Organization")])  # longer -> new surface, later batch
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f2], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            name2, aliases2 = _canonical_row(cur, "organization:acme")
        # the longer surface from the LATER batch now wins the display name...
        assert name2 == "Acme Corporation"
        # ...and the founding batch's surface is preserved as an alias, not dropped.
        assert aliases2 == ["Acme Corp"]
    finally:
        _cleanup(conn, [f1, f2], ["organization:acme"])


@pytest.mark.ac("KG-AC-78")
def test_attributes_merge_with_conflict_retention_across_documents(conn):
    fa = f"canon-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, fa, [
                ("Acme Corp", "Organization", "docA",
                 [{"property": "governingLaw", "value": "England and Wales",
                   "normalized_value": "England and Wales", "evidence": "e1",
                   "source_doc_id": "docA", "page": 1}]),
                ("Acme Corp", "Organization", "docB",
                 [{"property": "governingLaw", "value": "England and Wales",
                   "normalized_value": "England and Wales", "evidence": "e2",
                   "source_doc_id": "docB", "page": 3},
                  {"property": "effectiveDate", "value": "20 March 2025",
                   "normalized_value": "2025-03-20", "evidence": "e3",
                   "source_doc_id": "docB", "page": 1}]),
            ])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [fa], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            attrs = _canonical_attributes(cur, "organization:acme")
        # agreeing fact collapses, BOTH sources present
        assert len(attrs["governingLaw"]) == 1
        assert attrs["governingLaw"][0]["conflicting"] is False
        docs = {p["source_doc_id"] for p in attrs["governingLaw"][0]["provenance"]}
        assert docs == {"docA", "docB"}
        # single-mention property still merges cleanly (no conflict, one source)
        assert len(attrs["effectiveDate"]) == 1
        assert attrs["effectiveDate"][0]["conflicting"] is False
    finally:
        _cleanup(conn, [fa], ["organization:acme"])


@pytest.mark.ac("KG-AC-78")
def test_conflicting_attributes_across_a_cross_run_merge(conn):
    # KG-AC-78's own "never last-write-wins" promise must hold across batches too, not just
    # within one -- a later document disagreeing with an earlier one is a finding, not overwritten.
    f1, f2 = f"canon-{uuid.uuid4()}", f"canon-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            _seed(cur, f1, [("Acme Corp", "Organization", "docA",
                            [{"property": "effectiveDate", "value": "15 March 2025",
                              "normalized_value": "2025-03-15", "evidence": "e1",
                              "source_doc_id": "docA", "page": 1}])])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f1], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            _seed(cur, f2, [("Acme Corporation", "Organization", "docB",
                            [{"property": "effectiveDate", "value": "20 March 2025",
                              "normalized_value": "2025-03-20", "evidence": "e2",
                              "source_doc_id": "docB", "page": 1}])])
        conn.commit()
        canonicalize_batch(_DbShim(conn), [f2], pack=FIBO)
        conn.commit()

        with conn.cursor() as cur:
            attrs = _canonical_attributes(cur, "organization:acme")
        entries = attrs["effectiveDate"]
        assert len(entries) == 2  # NEITHER document's value overwritten
        assert all(e["conflicting"] is True for e in entries)
        assert {e["normalized_value"] for e in entries} == {"2025-03-15", "2025-03-20"}
    finally:
        _cleanup(conn, [f1, f2], ["organization:acme"])


@pytest.mark.ac("KG-AC-79")
def test_resolve_or_mint_collision_gets_a_deterministic_suffix_not_a_wrong_reuse(conn):
    # slugify collapses ANY run of non-alnum to one "-", so two genuinely DIFFERENT
    # normalized_form values can slug identically (e.g. differing only in punctuation style) even
    # though normalize_surface's own output rarely produces this in practice. _resolve_or_mint must
    # retry with KG-AC-79's suffix rather than silently reusing the FIRST cluster's canonical_id for
    # a SECOND, unrelated real-world entity -- exactly the defect this AC's collision rule exists
    # to prevent.
    from store import _resolve_or_mint
    try:
        with conn.cursor() as cur:
            cid1, minted1 = _resolve_or_mint(cur, "Organization", "acme!!!corp")
            cid2, minted2 = _resolve_or_mint(cur, "Organization", "acme---corp")
        conn.commit()
        assert minted1 is True and minted2 is True  # BOTH are novel mints, not a false merge
        assert cid1 != cid2  # two DISTINCT real clusters, never collapsed into one

        with conn.cursor() as cur:
            cur.execute(
                "SELECT canonical_key FROM public.kg_canonical_entities WHERE canonical_id IN (%s,%s) ORDER BY canonical_key",
                (cid1, cid2),
            )
            keys = [r[0] for r in cur.fetchall()]
        assert keys == ["organization:acme-corp", "organization:acme-corp-1"]

        # re-resolving EITHER original normalized_form must land on its OWN existing row, not mint
        # a third row and not cross-wire to the other cluster.
        with conn.cursor() as cur:
            cid1_again, minted_again = _resolve_or_mint(cur, "Organization", "acme!!!corp")
        conn.commit()
        assert minted_again is False and cid1_again == cid1
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM public.kg_canonical_entities WHERE canonical_key IN (%s,%s)",
                ("organization:acme-corp", "organization:acme-corp-1"),
            )
        conn.commit()


@pytest.mark.ac("KG-AC-40")
def test_mid_batch_failure_rolls_back_all_staged(conn):
    f = f"canon-{uuid.uuid4()}"
    try:
        with conn.cursor() as cur:
            # an ambiguous pair (fuzzy band) forces the adjudicator to run
            _seed(cur, f, [("Acme Systems", "Organization"), ("Acme Solutions", "Organization")])
        conn.commit()

        def boom(a, b):
            raise RuntimeError("adjudication failure mid-batch")

        with pytest.raises(RuntimeError):
            canonicalize_batch(_DbShim(conn), [f], fuzzy_floor=0.5, fuzzy_ceiling=0.99, pack=FIBO, adjudicate=boom)
        conn.rollback()  # the worker wrapper rolls back on failure

        with conn.cursor() as cur:
            rows = _rows(cur, f)
        # atomic: nothing canonicalized — all still staged, no canonical_id
        assert all(r[3] == "staged" and r[2] is None for r in rows)
    finally:
        _cleanup(conn, [f])
