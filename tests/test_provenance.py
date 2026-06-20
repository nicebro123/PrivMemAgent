import stat

from src.policy import RouteAction
from src.provenance import ProvenanceRecord, ProvenanceStore


def test_provenance_supports_cloud_ids_and_revocation(tmp_path):
    store = ProvenanceStore(str(tmp_path / "provenance.db"))
    record = ProvenanceRecord(
        record_id="r1",
        user_id="u1",
        source_message_id="m1",
        source_item_index=0,
        policy_version="v1",
        route_action=RouteAction.PUBLIC_ABSTRACT,
        rule_id="level:PL2",
        privacy_level="PL2",
        privacy_type="Email",
        representation_type="category_abstract",
        public_text="the user's contact information",
        alias_scope=None,
        scope_id=None,
        created_at="2026-06-20T00:00:00+00:00",
    )
    store.add(record)
    store.attach_cloud_memory_id("r1", "cloud-1")

    cloud_ids = store.revoke_by_source("u1", ["m1"], "forget request")
    updated = store.get("r1")

    assert cloud_ids == ["cloud-1"]
    assert updated.active is False
    assert updated.revocation_reason == "forget request"
    assert store.list_active("u1") == []
    store.close()
    assert stat.S_IMODE((tmp_path / "provenance.db").stat().st_mode) == 0o600
