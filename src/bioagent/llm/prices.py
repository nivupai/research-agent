"""List prices (USD per 1M tokens) used to estimate run cost.

Free-tier Gemini calls cost nothing, but we log the paid-tier list-price equivalent so runs
can be compared. Source: https://aistudio.google.com/docs/pricing (read 2026-11-12; output
price includes thinking tokens). Models missing here have unknown cost (reported as None).
Local Ollama models cost 0.
"""

from __future__ import annotations

PRICES_USD_PER_1M: dict[str, tuple[float, float]] = {
    # model: (input, output)
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.8-flash": (0.75, 3.75),  # through 2026-12-31; doubles from 2027-01-01
}


def estimate_cost(provider: str, model: str, input_tokens: int, output_tokens: int) -> float | None:
    if provider == "ollama":
        return 0.0
    price = PRICES_USD_PER_1M.get(model)
    if price is None:
        return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 6)
