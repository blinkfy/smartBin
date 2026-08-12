from __future__ import annotations

from build_paper_motivation_decision_packet import (
    _option_definitions,
    build_packet,
)


def test_motivation_options_have_one_recommendation() -> None:
    options = _option_definitions()

    assert [option["id"] for option in options] == ["A", "B", "C"]
    assert sum(option["recommended"] for option in options) == 1
    assert next(
        option["id"] for option in options if option["recommended"]
    ) == "A"
    assert all(option["results_validation"] for option in options)


def test_decision_packet_does_not_fake_user_confirmation(tmp_path) -> None:
    output = tmp_path / "MOTIVATION_DECISION_PACKET.json"

    packet = build_packet(output)

    assert packet["valid"] is True
    assert packet["state"] == "awaiting_user_confirmation"
    assert packet["confirmed_option"] is None
    assert packet["confirmation_required"] is True
    assert packet["paper_spine_gate"][
        "substantive_full_drafting_authorized"
    ] is False
