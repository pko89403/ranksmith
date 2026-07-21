from __future__ import annotations


def validate_no_answer_value(value: object) -> None:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError("no_answer_value must be a non-empty string")
