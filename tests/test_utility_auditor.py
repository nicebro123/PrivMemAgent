import stat

from src.alias_router import ScopedAliasRouter
from src.policy import PrivacyPolicy, RoutingContext
from src.provenance import ProvenanceStore
from src.public_memory_compiler import PublicMemoryCompiler
from src.utility_auditor import UtilityProxyAuditor


def test_utility_proxy_reports_local_private_recoverability(tmp_path):
    router = ScopedAliasRouter(
        str(tmp_path / "aliases.db"),
        key_path=str(tmp_path / "aliases.key"),
    )
    compiler = PublicMemoryCompiler(
        PrivacyPolicy.default(),
        router,
        ProvenanceStore(str(tmp_path / "provenance.db")),
    )
    item = {
        "original_text": "alice@example.com",
        "privacy_type": "Email",
        "privacy_level": "PL2",
    }
    record = compiler.compile(
        "My email is alice@example.com.",
        [item],
        RoutingContext(
            user_id="u1",
            message_id="m1",
            turn_id="t1",
            session_id="s1",
            task_id="task1",
        ),
    )
    report = UtilityProxyAuditor().audit(
        [record],
        {
            "u1": [
                {
                    "answer": "alice@example.com",
                    "evidence": "The email is alice@example.com.",
                }
            ]
        },
        {"u1": [item]},
        router,
    )

    assert report.oracle_type_local_recoverability == 1.0
    assert report.local_recoverability_applicable is True
    assert report.pl4_local_retention_rate == 0.0
    assert report.proxy_only is True
    compiler.close()
    assert stat.S_IMODE((tmp_path / "aliases.key").stat().st_mode) == 0o600


def test_utility_proxy_does_not_report_vacuous_private_recall(tmp_path):
    router = ScopedAliasRouter(
        str(tmp_path / "aliases.db"),
        key_path=str(tmp_path / "aliases.key"),
    )
    report = UtilityProxyAuditor().audit(
        [],
        {"u1": [{"answer": "No private value is referenced.", "evidence": ""}]},
        {"u1": []},
        router,
    )

    assert report.oracle_type_local_recoverability is None
    assert report.local_recoverability_applicable is False


def test_utility_proxy_uses_highest_level_for_conflicting_labels(tmp_path):
    router = ScopedAliasRouter(
        str(tmp_path / "aliases.db"),
        key_path=str(tmp_path / "aliases.key"),
    )
    report = UtilityProxyAuditor().audit(
        [],
        {"u1": [{"answer": "shared-secret", "evidence": ""}]},
        {
            "u1": [
                {
                    "original_text": "shared-secret",
                    "privacy_type": "Account ID",
                    "privacy_level": "PL2",
                },
                {
                    "original_text": "shared-secret",
                    "privacy_type": "Password",
                    "privacy_level": "PL4",
                },
            ]
        },
        router,
    )

    assert report.exactly_referenced_private_facts == 1
    assert report.policy_eligible_private_facts == 0
    assert report.oracle_type_local_recoverability is None



def test_utility_proxy_resolves_multiple_choice_answer_labels(tmp_path):
    router = ScopedAliasRouter(
        str(tmp_path / "aliases.db"),
        key_path=str(tmp_path / "aliases.key"),
    )
    compiler = PublicMemoryCompiler(
        PrivacyPolicy.default(),
        router,
        ProvenanceStore(str(tmp_path / "provenance.db")),
    )
    record = compiler.compile(
        "The user likes naturally leavened bread with heritage grains.",
        [],
        RoutingContext(
            user_id="u1",
            message_id="m1",
            turn_id="t1",
            session_id="s1",
            task_id="task1",
        ),
    )

    report = UtilityProxyAuditor().audit(
        [record],
        {
            "u1": [
                {
                    "answer": "(c)",
                    "all_options": [
                        "(a) The user prefers painting landscapes.",
                        "(b) The user prefers woodworking projects.",
                        "(c) The user likes naturally leavened bread with heritage grains.",
                    ],
                }
            ]
        },
        {"u1": []},
        router,
    )

    assert report.non_private_answer_token_recall == 1.0
    compiler.close()


def test_utility_proxy_detects_private_fact_inside_selected_option(tmp_path):
    router = ScopedAliasRouter(
        str(tmp_path / "aliases.db"),
        key_path=str(tmp_path / "aliases.key"),
    )
    compiler = PublicMemoryCompiler(
        PrivacyPolicy.default(),
        router,
        ProvenanceStore(str(tmp_path / "provenance.db")),
    )
    item = {
        "original_text": "+1-617-492-7843",
        "privacy_type": "Phone Number",
        "privacy_level": "PL2",
    }
    record = compiler.compile(
        "My phone number is +1-617-492-7843.",
        [item],
        RoutingContext(
            user_id="u1",
            message_id="m1",
            turn_id="t1",
            session_id="s1",
            task_id="task1",
        ),
    )

    report = UtilityProxyAuditor().audit(
        [record],
        {
            "u1": [
                {
                    "answer": "(a)",
                    "all_options": [
                        "(a) +1-617-492-7843",
                        "(b) +1-617-294-7843",
                    ],
                }
            ]
        },
        {"u1": [item]},
        router,
    )

    assert report.exactly_referenced_private_facts == 1
    assert report.oracle_type_local_recoverability == 1.0
    compiler.close()
