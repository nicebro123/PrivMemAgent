from src.alias_router import ScopedAliasRouter
from src.leakage_auditor import AuditThresholds, LeakageAuditor
from src.policy import AliasScope, PrivacyPolicy, RouteAction, RoutingContext
from src.provenance import ProvenanceStore
from src.public_memory_compiler import CompiledMemory, PublicMemoryCompiler
from src.query_broker import EdgeQueryBroker


def make_compiler(tmp_path):
    router = ScopedAliasRouter(
        str(tmp_path / "aliases.db"),
        key_path=str(tmp_path / "aliases.key"),
    )
    provenance = ProvenanceStore(str(tmp_path / "provenance.db"))
    return PublicMemoryCompiler(
        policy=PrivacyPolicy.default(),
        alias_router=router,
        provenance_store=provenance,
    )


def context(message_id="m1", task_id="task-1", **kwargs):
    values = {
        "user_id": "u1",
        "message_id": message_id,
        "turn_id": f"turn-{message_id}",
        "session_id": "session-1",
        "task_id": task_id,
    }
    values.update(kwargs)
    return RoutingContext(**values)


def test_compiler_routes_pl2_pl3_pl4_without_exact_leakage(tmp_path):
    compiler = make_compiler(tmp_path)
    text = "Email alice@example.com. Diagnosis severe anxiety. Verification code 829417."
    items = [
        {
            "original_text": "alice@example.com",
            "privacy_type": "Email",
            "privacy_level": "PL2",
        },
        {
            "original_text": "severe anxiety",
            "privacy_type": "Medical Health",
            "privacy_level": "PL3",
        },
        {
            "original_text": "829417",
            "privacy_type": "Verification Code",
            "privacy_level": "PL4",
        },
    ]

    compiled = compiler.compile(text, items, context())

    assert "alice@example.com" not in compiled.public_text
    assert "severe anxiety" not in compiled.public_text
    assert "829417" not in compiled.public_text
    assert "contact information" in compiled.public_text
    assert [item.decision.action for item in compiled.items] == [
        RouteAction.PUBLIC_ABSTRACT,
        RouteAction.LOCAL_ONLY,
        RouteAction.DROP,
    ]
    assert len(compiler.provenance_store.list_active("u1")) == 3
    compiler.close()


def test_category_abstraction_avoids_duplicate_articles(tmp_path):
    compiler = make_compiler(tmp_path)
    compiled = compiler.compile(
        "I work as a professor.",
        [
            {
                "original_text": "professor",
                "privacy_type": "Job Information",
                "privacy_level": "PL2",
            }
        ],
        context(),
    )

    assert compiled.public_text == "I work as a professional or educational context."
    compiler.close()


def test_task_scoped_alias_is_stable_inside_task_and_rotates_across_tasks(tmp_path):
    compiler = make_compiler(tmp_path)
    item = {
        "original_text": "alice@example.com",
        "privacy_type": "Email",
        "privacy_level": "PL2",
    }
    consent = {
        "exact_required_types": {"Email"},
        "consented_reversible_types": {"Email"},
    }
    first = compiler.compile(
        "Email alice@example.com",
        [item],
        context("m1", task_id="task-a", **consent),
    )
    repeated = compiler.compile(
        "Email alice@example.com",
        [item],
        context("m2", task_id="task-a", **consent),
    )
    rotated = compiler.compile(
        "Email alice@example.com",
        [item],
        context("m3", task_id="task-b", **consent),
    )

    first_alias = first.items[0].public_value
    assert first_alias == repeated.items[0].public_value
    assert first_alias != rotated.items[0].public_value
    assert first.items[0].alias_scope == AliasScope.TASK

    restored = compiler.alias_router.restore(
        f"Send to {first_alias}",
        context("m4", task_id="task-a"),
    )
    assert restored == "Send to alice@example.com"
    compiler.close()


def test_auditor_measures_exact_recovery_pl4_and_scope_linkability(tmp_path):
    compiler = make_compiler(tmp_path)
    item = {
        "original_text": "alice@example.com",
        "privacy_type": "Email",
        "privacy_level": "PL2",
    }
    consent = {
        "exact_required_types": {"Email"},
        "consented_reversible_types": {"Email"},
    }
    records = [
        compiler.compile(
            "Email alice@example.com",
            [item],
            context("m1", task_id="task-a", **consent),
        ),
        compiler.compile(
            "Email alice@example.com",
            [item],
            context("m2", task_id="task-b", **consent),
        ),
    ]
    auditor = LeakageAuditor(
        AuditThresholds(
            exact_recovery=0.0,
            cross_scope_linkability=0.0,
            pl4_public_retention=0.0,
            minimum_token_reduction=-1.0,
        )
    )
    report = auditor.audit(
        records,
        {
            "m1": [item],
            "m2": [item],
        },
    )

    assert report.exact_recovery_rate == 0.0
    assert report.cross_scope_linkability_rate == 0.0
    assert report.cross_scope_linkability_applicable is True
    assert report.passed is True
    compiler.close()


def test_token_reduction_gate_uses_corpus_weighting():
    records = [
        CompiledMemory(
            user_id="u1",
            message_id="large",
            source_fingerprint="a",
            public_text="",
            items=(),
            policy_version="v1",
            source_tokens=100,
            public_tokens=50,
        ),
        CompiledMemory(
            user_id="u1",
            message_id="small",
            source_fingerprint="b",
            public_text="",
            items=(),
            policy_version="v1",
            source_tokens=1,
            public_tokens=1,
        ),
    ]
    report = LeakageAuditor(
        AuditThresholds(minimum_token_reduction=0.30)
    ).audit(records, {})

    assert report.average_token_reduction == 0.25
    assert report.corpus_token_reduction > 0.49
    assert report.passed is True


def test_edge_query_broker_hydrates_aliases_and_restores_response(tmp_path):
    compiler = make_compiler(tmp_path)
    item = {
        "original_text": "alice@example.com",
        "privacy_type": "Email",
        "privacy_level": "PL2",
    }
    compiler.compile(
        "Email alice@example.com",
        [item],
        context("m1", task_id="ingestion"),
    )
    broker = EdgeQueryBroker(compiler.alias_router)
    query_context = context(
        "q1",
        task_id="ticket-delivery",
        exact_required_types={"Email"},
        consented_reversible_types={"Email"},
    )
    prepared = broker.prepare(
        "Which address should receive the ticket?",
        ["Email"],
        query_context,
    )

    assert "alice@example.com" not in prepared.cloud_query
    assert len(prepared.aliases) == 1
    alias = prepared.aliases[0].alias
    assert alias in prepared.cloud_query
    assert (
        broker.restore_response(
            f"Send the ticket to {alias}.",
            query_context,
            prepared,
        )
        == "Send the ticket to alice@example.com."
    )
    compiler.close()


def test_edge_query_broker_rejects_unauthorized_private_hydration(tmp_path):
    compiler = make_compiler(tmp_path)
    broker = EdgeQueryBroker(compiler.alias_router)

    try:
        broker.prepare(
            "Which address should receive the ticket?",
            ["Email"],
            context("q1", task_id="ticket-delivery"),
        )
    except PermissionError as error:
        assert "Email" in str(error)
    else:
        raise AssertionError("unauthorized private hydration must fail closed")
    compiler.close()


def test_public_memory_compiler_drops_entire_secret_sentence_for_pl4(tmp_path):
    from src.alias_router import ScopedAliasRouter
    from src.policy import PrivacyPolicy, RoutingContext
    from src.provenance import ProvenanceStore
    from src.public_memory_compiler import PublicMemoryCompiler

    compiler = PublicMemoryCompiler(
        policy=PrivacyPolicy.default(),
        alias_router=ScopedAliasRouter(
            str(tmp_path / "aliases.db"),
            key_path=str(tmp_path / "aliases.key"),
        ),
        provenance_store=ProvenanceStore(str(tmp_path / "provenance.db")),
    )

    compiled = compiler.compile(
        message_text=(
            "My favorite weekend activity is birdwatching. "
            "The temporary verification code is 829417. "
            "Please remember the hobby."
        ),
        privacy_items=[
            {
                "original_text": "829417",
                "privacy_type": "Verification Code",
                "privacy_level": "PL4",
            }
        ],
        context=RoutingContext(user_id="u1", message_id="m1"),
    )

    assert "829417" not in compiled.public_text
    assert "verification code" not in compiled.public_text.lower()
    assert "birdwatching" in compiled.public_text


def test_public_memory_compiler_drops_assistant_privacy_boilerplate(tmp_path):
    from src.alias_router import ScopedAliasRouter
    from src.policy import PrivacyPolicy, RoutingContext
    from src.provenance import ProvenanceStore
    from src.public_memory_compiler import PublicMemoryCompiler

    compiler = PublicMemoryCompiler(
        policy=PrivacyPolicy.default(),
        alias_router=ScopedAliasRouter(
            str(tmp_path / "aliases.db"),
            key_path=str(tmp_path / "aliases.key"),
        ),
        provenance_store=ProvenanceStore(str(tmp_path / "provenance.db")),
    )

    compiled = compiler.compile(
        message_text="I will not retain the verification code.",
        privacy_items=[],
        context=RoutingContext(
            user_id="u1",
            message_id="m1",
            message_role="assistant",
        ),
    )

    assert compiled.public_text == ""


def test_public_memory_compiler_merges_overlapping_local_and_drop_ranges(tmp_path):
    from src.alias_router import ScopedAliasRouter
    from src.policy import PrivacyPolicy, RoutingContext
    from src.provenance import ProvenanceStore
    from src.public_memory_compiler import PublicMemoryCompiler

    compiler = PublicMemoryCompiler(
        policy=PrivacyPolicy.default(),
        alias_router=ScopedAliasRouter(
            str(tmp_path / "aliases.db"),
            key_path=str(tmp_path / "aliases.key"),
        ),
        provenance_store=ProvenanceStore(str(tmp_path / "provenance.db")),
    )

    compiled = compiler.compile(
        message_text=(
            "I purchased a report for $150.00 for it, and the transaction ID "
            "is TXN-88492-TAS. I need to check the land."
        ),
        privacy_items=[
            {
                "original_text": "report for $150.00 for it",
                "privacy_type": "Transaction Record",
                "privacy_level": "PL3",
            },
            {
                "original_text": "TXN-88492-TAS",
                "privacy_type": "Project/Task ID",
                "privacy_level": "PL4",
            },
        ],
        context=RoutingContext(user_id="u1", message_id="m1"),
    )

    assert "TXN-88492-TAS" not in compiled.public_text
    assert "transaction ID" not in compiled.public_text
    assert "check the land" in compiled.public_text


def test_public_memory_compiler_drops_entire_local_only_sentence_for_pl3(tmp_path):
    from src.alias_router import ScopedAliasRouter
    from src.policy import PrivacyPolicy, RoutingContext
    from src.provenance import ProvenanceStore
    from src.public_memory_compiler import PublicMemoryCompiler

    compiler = PublicMemoryCompiler(
        policy=PrivacyPolicy.default(),
        alias_router=ScopedAliasRouter(
            str(tmp_path / "aliases.db"),
            key_path=str(tmp_path / "aliases.key"),
        ),
        provenance_store=ProvenanceStore(str(tmp_path / "provenance.db")),
    )

    compiled = compiler.compile(
        message_text=(
            "The restaurant should be quiet. "
            "My Amex card ending in 8865 paid for the trip. "
            "I prefer window seats."
        ),
        privacy_items=[
            {
                "original_text": "card ending in 8865",
                "privacy_type": "Financial Account",
                "privacy_level": "PL3",
            }
        ],
        context=RoutingContext(user_id="u1", message_id="m1"),
    )

    assert "Amex" not in compiled.public_text
    assert "8865" not in compiled.public_text
    assert "window seats" in compiled.public_text


def test_public_memory_compiler_drops_user_forget_control_messages(tmp_path):
    from src.alias_router import ScopedAliasRouter
    from src.policy import PrivacyPolicy, RoutingContext
    from src.provenance import ProvenanceStore
    from src.public_memory_compiler import PublicMemoryCompiler

    compiler = PublicMemoryCompiler(
        policy=PrivacyPolicy.default(),
        alias_router=ScopedAliasRouter(
            str(tmp_path / "aliases.db"),
            key_path=str(tmp_path / "aliases.key"),
        ),
        provenance_store=ProvenanceStore(str(tmp_path / "provenance.db")),
    )

    compiled = compiler.compile(
        message_text="Please forget that I enjoy casual cycling around the neighborhood.",
        privacy_items=[],
        context=RoutingContext(user_id="u1", message_id="m1"),
    )

    assert compiled.public_text == ""
    compiler.close()


def test_public_memory_compiler_drops_ephemeral_task_requests(tmp_path):
    from src.alias_router import ScopedAliasRouter
    from src.policy import PrivacyPolicy, RoutingContext
    from src.provenance import ProvenanceStore
    from src.public_memory_compiler import PublicMemoryCompiler

    compiler = PublicMemoryCompiler(
        policy=PrivacyPolicy.default(),
        alias_router=ScopedAliasRouter(
            str(tmp_path / "aliases.db"),
            key_path=str(tmp_path / "aliases.key"),
        ),
        provenance_store=ProvenanceStore(str(tmp_path / "provenance.db")),
    )

    compiled = compiler.compile(
        message_text="Could you also suggest a few alternative metaphors for this paragraph?",
        privacy_items=[],
        context=RoutingContext(user_id="u1", message_id="m1"),
    )

    assert compiled.public_text == ""
    compiler.close()
