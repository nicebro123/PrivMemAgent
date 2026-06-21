from __future__ import annotations

from src.abstraction_generator import (
    AbstractionCandidate,
    AbstractionInput,
    AbstractorAdapter,
)
from src.alias_router import ScopedAliasRouter
from src.policy import PrivacyPolicy, RoutingContext
from src.provenance import ProvenanceStore
from src.public_memory_compiler import PublicMemoryCompiler
from src.sufficiency_selector import SelectorConfig
from src.utility_leakage_selector import LearnedUtilityLeakageSelector, LinearUtilityLeakageRanker


class DemoGenerator:
    def generate(self, item: AbstractionInput):
        return [
            AbstractionCandidate(
                text="private detail",
                abstraction_level=1,
                representation_type="generic_abstract",
                utility_score=0.70,
                leakage_score=0.05,
            ),
            AbstractionCandidate(
                text="contact information",
                abstraction_level=2,
                representation_type="learned_contact_abstract",
                utility_score=0.95,
                leakage_score=0.20,
            ),
        ]


def make_compiler(tmp_path):
    return PublicMemoryCompiler(
        policy=PrivacyPolicy.default(),
        alias_router=ScopedAliasRouter(
            str(tmp_path / "aliases.db"),
            key_path=str(tmp_path / "aliases.key"),
        ),
        provenance_store=ProvenanceStore(str(tmp_path / "provenance.db")),
        abstractor=AbstractorAdapter(DemoGenerator()),
        learned_selector=LearnedUtilityLeakageSelector(
            config=SelectorConfig(utility_floor=0.5, max_leakage=0.35, max_public_tokens=20),
            ranker=LinearUtilityLeakageRanker({"utility_score": 4.0, "leakage_score": -1.0}),
        ),
    )


def context():
    return RoutingContext(
        user_id="u1",
        message_id="m1",
        turn_id="turn-1",
        session_id="session-1",
        task_id="task-1",
    )


def test_public_memory_compiler_can_use_learned_modules(tmp_path):
    compiler = make_compiler(tmp_path)

    compiled = compiler.compile(
        "Email alice@example.com.",
        [
            {
                "original_text": "alice@example.com",
                "privacy_type": "Email",
                "privacy_level": "PL2",
            }
        ],
        context(),
    )

    assert compiled.items[0].representation_type == "learned_contact_abstract"
    assert compiled.items[0].public_value == "contact information"
    assert "alice@example.com" not in compiled.public_text
    assert "contact information" in compiled.public_text
    compiler.close()
