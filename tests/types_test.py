"""Invariants of the shared type definitions in ``opengradient.types``."""

from opengradient.types import TEE_LLM


def test_tee_llm_values_are_provider_prefixed():
    """Every model id is ``provider/model``.

    The clients send only the part after the slash to the TEE gateway
    (``model.split("/")[1]``), so a value without a prefix would send the
    provider name as the model and be rejected by the gateway.
    """
    for member in TEE_LLM:
        provider, _, model_id = member.value.partition("/")
        assert provider, f"{member.name} has no provider prefix"
        assert model_id, f"{member.name} has no model id"
        assert "/" not in model_id, f"{member.name} has more than one slash"


def test_tee_llm_gateway_ids_are_unique():
    """No two members resolve to the same gateway model id.

    ``TEE_LLM`` is a ``str`` enum, so two members sharing a *value* would
    silently alias; two members with different prefixes but the same model id
    would instead be two names for one gateway model.
    """
    seen: dict[str, str] = {}
    for member in TEE_LLM:
        model_id = member.value.split("/", 1)[1]
        assert model_id not in seen, f"{member.name} duplicates {seen[model_id]} ({model_id})"
        seen[model_id] = member.name


def test_tee_llm_has_no_aliased_members():
    """Duplicate enum values collapse into aliases — catch that explicitly."""
    assert len(list(TEE_LLM)) == len(TEE_LLM.__members__)
