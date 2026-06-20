from src.context_minimizer import ContextMinimizer, ContextMinimizerConfig


def test_minimizer_removes_boilerplate_and_keeps_protected_fact():
    minimizer = ContextMinimizer(
        ContextMinimizerConfig(
            target_public_ratio=0.5,
            max_public_tokens=30,
            min_public_tokens=3,
        )
    )
    text = (
        "Of course, happy to help. "
        "The user's contact information is required for ticket delivery. "
        "Would you like me to suggest anything else?"
    )
    result = minimizer.minimize(
        text,
        role="assistant",
        protected_fragments=["the user's contact information"],
    )

    assert "contact information" in result.text
    assert "happy to help" not in result.text
    assert result.public_tokens < result.source_tokens


def test_minimizer_preserves_original_order_after_ranking():
    minimizer = ContextMinimizer(
        ContextMinimizerConfig(
            target_public_ratio=0.8,
            max_public_tokens=30,
            min_public_tokens=3,
        )
    )
    text = "I prefer tea. Here are unrelated suggestions. I now live in Paris."
    result = minimizer.minimize(text, role="user")
    assert result.text.index("prefer tea") < result.text.index("live in Paris")


def test_minimizer_preserves_fact_when_protected_detail_uses_budget():
    minimizer = ContextMinimizer(
        ContextMinimizerConfig(
            target_public_ratio=0.40,
            max_public_tokens=64,
            min_public_tokens=4,
        )
    )
    result = minimizer.minimize(
        "My preferred quiet activity is birdwatching. "
        "My email is contact information.",
        role="user",
        protected_fragments=["contact information"],
    )

    assert "birdwatching" in result.text
    assert "contact information" in result.text


def test_minimizer_preserves_trailing_quote():
    minimizer = ContextMinimizer()
    text = 'The user said: "My preferred activity is birdwatching."'

    assert minimizer.minimize(text, role="user").text == text
