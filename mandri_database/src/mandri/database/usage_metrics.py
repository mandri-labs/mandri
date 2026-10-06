import json
from collections import Counter
from decimal import Context, Decimal
from typing import Any

from mandri.database.usage_serialization import COUNTERS


class UsageMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.totals = dict.fromkeys(COUNTERS, 0)
        self.missing = dict.fromkeys(COUNTERS, 0)
        self.amounts: list[Decimal | None] = [None, None]
        self._contexts = [Context(prec=28), Context(prec=28)]
        self._integer_digits = [0, 0]
        self._fraction_digits = [0, 0]
        self.unpriced = 0
        self.incomplete = 0
        self.unclassified = 0
        self.bases: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()

    def step(self, *values: Any) -> None:
        self.count += 1
        for key, value in zip(COUNTERS, values[:7], strict=True):
            if value is None:
                self.missing[key] += 1
            else:
                self.totals[key] += int(value)
        for index, value in enumerate(values[7:9]):
            if value is not None:
                self._add_amount(index, value)
        complete, model, basis = values[9:12]
        self.incomplete += not complete
        self.unclassified += not model
        self.bases[basis or "unpriced"] += 1
        if values[7] is None:
            self.unpriced += 1
            reason = (
                "model_missing"
                if not model
                else "token_usage_missing"
                if values[0] is None or values[1] is None
                else "tariff_or_context_unavailable"
            )
            self.reasons[reason] += 1

    def _add_amount(self, index: int, value: str) -> None:
        amount = Decimal(value)
        previous = self.amounts[index] or Decimal(0)
        self._integer_digits[index] = max(self._integer_digits[index], amount.adjusted() + 1)
        self._fraction_digits[index] = max(
            self._fraction_digits[index], -int(amount.as_tuple().exponent)
        )
        context = self._contexts[index]
        context.prec = max(
            28,
            self._integer_digits[index] + self._fraction_digits[index] + len(str(self.count)) + 1,
        )
        self.amounts[index] = context.add(previous, amount)

    def result(self) -> dict[str, Any]:
        return {
            "fact_count": self.count,
            **{
                key: value if self.missing[key] < self.count else None
                for key, value in self.totals.items()
            },
            "missing_fields": self.missing,
            "usd_equivalent": str(self.amounts[0]) if self.amounts[0] is not None else None,
            "reported_cost_usd": str(self.amounts[1]) if self.amounts[1] is not None else None,
            "unpriced_fact_count": self.unpriced,
            "incomplete_fact_count": self.incomplete,
            "unclassified_fact_count": self.unclassified,
            "valuation_bases": dict(self.bases),
            "unpriced_reasons": dict(self.reasons),
        }

    def finalize(self) -> str:
        return json.dumps(self.result(), separators=(",", ":"))


class MergedUsageMetrics(UsageMetrics):
    def step(self, *values: Any) -> None:
        value = json.loads(values[0])
        self.count += value["fact_count"]
        for key in COUNTERS:
            self.totals[key] += value[key] or 0
            self.missing[key] += value["missing_fields"][key]
        for index, field in enumerate(("usd_equivalent", "reported_cost_usd")):
            if value[field] is not None:
                self._add_amount(index, value[field])
        self.unpriced += value["unpriced_fact_count"]
        self.incomplete += value["incomplete_fact_count"]
        self.unclassified += value["unclassified_fact_count"]
        self.bases.update(value["valuation_bases"])
        self.reasons.update(value["unpriced_reasons"])
