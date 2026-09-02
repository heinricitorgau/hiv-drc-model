"""Download the DRC HIV indicators this package can fit, from the World Bank API.

The World Bank republishes UNAIDS' country estimates through a plain,
unauthenticated JSON API, which is the only machine-readable route to them
that does not involve scraping a JavaScript dashboard. AIDSinfo itself is a
single-page app; its numbers are the same UNAIDS estimates.

Indicators, and how they map onto the model:

===================  ==================================================
``SH.DYN.AIDS``      Adults (15+) living with HIV, a count -> ``plhiv``
``SH.HIV.ARTC.ZS``   ART coverage, % of PLHIV        -> ``art_coverage``
``SP.POP.TOTL``      Total population, a count       -> ``population``
===================  ==================================================

Writes a tidy CSV to ``data/real/drc_worldbank.csv`` with one row per year.
Network access happens here and nowhere else, so the loader that turns this
CSV into observations stays pure and testable offline.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.request
from pathlib import Path

INDICATORS = {
    "plhiv_adults": "SH.DYN.AIDS",
    "art_coverage_pct": "SH.HIV.ARTC.ZS",
    "population": "SP.POP.TOTL",
}
COUNTRY = "COD"  # Democratic Republic of the Congo
YEARS = "1990:2024"
BASE = "https://api.worldbank.org/v2"
OUT = Path("data/real/drc_worldbank.csv")


def fetch(indicator: str) -> dict[int, float | None]:
    """One indicator, as ``{year: value}``. ``None`` where the API reports null."""
    url = f"{BASE}/country/{COUNTRY}/indicator/{indicator}?format=json&per_page=500&date={YEARS}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
        payload = json.load(response)
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"unexpected response for {indicator}: {payload!r:.200}")
    return {int(row["date"]): row["value"] for row in payload[1]}


def main() -> int:
    series = {name: fetch(code) for name, code in INDICATORS.items()}
    for name, code in INDICATORS.items():
        present = sum(v is not None for v in series[name].values())
        print(f"  {code:18s} {name:18s} {present} of {len(series[name])} years present")

    years = sorted(set().union(*(set(s) for s in series.values())))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["year", *INDICATORS])
        for year in years:
            row = [year]
            for name in INDICATORS:
                value = series[name].get(year)
                row.append("" if value is None else f"{value:.10g}")
            writer.writerow(row)

    print(f"\nwrote {OUT} ({len(years)} rows, {YEARS})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
