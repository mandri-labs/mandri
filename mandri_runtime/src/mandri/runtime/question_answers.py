from typing import Any


def valid_question_answers(answers: list[dict[str, Any]] | None) -> bool:
    return bool(answers) and all(
        isinstance(item, dict)
        and isinstance(item.get("question"), str)
        and isinstance(item.get("answers"), list)
        and bool(item["answers"])
        and all(isinstance(answer, str) for answer in item["answers"])
        for item in answers or []
    )
