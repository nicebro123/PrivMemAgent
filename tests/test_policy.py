import pytest

from src.policy import (
    AliasScope,
    PrivacyPolicy,
    RouteAction,
    RoutingContext,
)


def context(**kwargs):
    defaults = {
        "user_id": "u1",
        "message_id": "m1",
        "turn_id": "turn-1",
        "session_id": "session-1",
        "task_id": "task-1",
    }
    defaults.update(kwargs)
    return RoutingContext(**defaults)


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("PL2", RouteAction.PUBLIC_ABSTRACT),
        ("PL3", RouteAction.LOCAL_ONLY),
        ("PL4", RouteAction.DROP),
    ],
)
def test_default_policy_routes_by_level(level, expected):
    decision = PrivacyPolicy.default().route(
        {
            "original_text": "secret",
            "privacy_type": "Email",
            "privacy_level": level,
        },
        context(),
    )
    assert decision.action == expected


def test_exact_value_requires_both_task_need_and_consent():
    item = {
        "original_text": "alice@example.com",
        "privacy_type": "Email",
        "privacy_level": "PL2",
    }
    no_consent = PrivacyPolicy.default().route(
        item,
        context(exact_required_types={"Email"}),
    )
    with_consent = PrivacyPolicy.default().route(
        item,
        context(
            exact_required_types={"Email"},
            consented_reversible_types={"Email"},
        ),
    )

    assert no_consent.action == RouteAction.LOCAL_ONLY
    assert with_consent.action == RouteAction.PUBLIC_REVERSIBLE
    assert with_consent.alias_scope == AliasScope.TASK


def test_scope_identifier_fails_closed_when_context_is_missing():
    with pytest.raises(ValueError, match="task_id"):
        context(task_id=None).scope_identifier(AliasScope.TASK)


def test_pl4_drop_cannot_be_overridden_by_need_or_consent():
    decision = PrivacyPolicy.default().route(
        {
            "original_text": "829417",
            "privacy_type": "Verification Code",
            "privacy_level": "PL4",
        },
        context(
            exact_required_types={"Verification Code"},
            consented_reversible_types={"Verification Code"},
        ),
    )

    assert decision.action == RouteAction.DROP
    assert decision.rule_id == "mandatory:PL4-drop"


def test_partial_alias_scope_config_preserves_other_defaults():
    policy = PrivacyPolicy.from_dict(
        {
            "public_memory": {
                "policy": {
                    "pl2": "public_abstract",
                    "pl3": "local_only",
                    "pl4": "drop",
                },
                "alias_scope": {"pl2": "persistent"},
            }
        }
    )
    decision = policy.route(
        {
            "original_text": "diagnosis",
            "privacy_type": "Medical Health",
            "privacy_level": "PL3",
        },
        context(
            exact_required_types={"Medical Health"},
            consented_reversible_types={"Medical Health"},
        ),
    )

    assert decision.alias_scope == AliasScope.SESSION
