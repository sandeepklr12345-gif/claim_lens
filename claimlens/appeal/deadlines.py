"""Ombudsman filing-window arithmetic (SPEC.md 14.4). Stdlib `datetime` only.

RULES were checked on 2026-11-13 against the text of Rule 14(3)(a) and (b) of the Insurance
Ombudsman Rules, 2017 in the copy "as amended till 09.11.2023" published by the Council for
Insurance Ombudsmen (cioins.co.in). Amendments after 9 Nov 2023 were NOT checked, so outputs
still tell the user to confirm with the Ombudsman office. The Rs 50 lakh award cap and other
rules are not used anywhere (not checked). Only the date arithmetic is unit-tested.
"""

from __future__ import annotations

import calendar
from datetime import date

from claimlens.schemas import DeadlineInfo

SOURCE = ("Insurance Ombudsman Rules, 2017, as amended till 09.11.2023, Council for Insurance "
          "Ombudsmen (cioins.co.in), read 2026-11-13")
RULES = [
    "Rule 14(3)(a): a complaint lies only after a written, e-mail or online representation to the "
    "insurer was rejected, got no reply within one month of receipt, or got an unsatisfactory "
    "reply.",
    "Rule 14(3)(b): the complaint must be made within one year after the rejection order is "
    "received, after an unsatisfactory decision is received, or after one month from sending "
    "the written representation.",
]
VERIFY_NOTE = ("Rule texts checked against the published copy as amended till 09.11.2023; later "
               "amendments were not checked. Confirm with the Ombudsman office before relying on "
               "a date.")
CONSERVATIVE_BASIS = (
    "No representation to the insurer is recorded. The latest date is a conservative outer bound "
    "counted from the date of rejection. Make a written representation to the insurer first."
)


def add_months(d: date, months: int) -> date:
    """Add calendar months, clamping the day to the target month's length.

    31 Jan + 1 month = 28/29 Feb; 29 Feb + 12 months = 28 Feb of the next year.
    """
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def parse_date(value: str | date | None, name: str) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise ValueError(f"{name} must be a date like 2026-03-31, got {value!r}") from e


def compute_deadlines(
    rejection_date: str | date,
    grievance_filed_date: str | date | None = None,
    insurer_reply_date: str | date | None = None,
) -> DeadlineInfo:
    """Earliest and latest Ombudsman complaint dates implied by the given events."""
    rej = parse_date(rejection_date, "rejection_date")
    gri = parse_date(grievance_filed_date, "grievance_filed_date")
    rep = parse_date(insurer_reply_date, "insurer_reply_date")
    if rej is None:
        raise ValueError("rejection_date is required")
    if gri and gri < rej:
        raise ValueError("grievance_filed_date is before rejection_date")
    if rep and rep < (gri or rej):
        raise ValueError("insurer_reply_date is before the grievance/rejection date")

    conservative = False
    if rep:
        earliest, basis = rep, "one year counted from the insurer's reply to the representation"
    elif gri:
        earliest = add_months(gri, 1)
        basis = ("no insurer reply recorded: complaint possible one month after the "
                 "representation; one year counted from that date")
    else:
        earliest, conservative, basis = None, True, CONSERVATIVE_BASIS
    latest = add_months(earliest or rej, 12)
    return DeadlineInfo(
        rejection_date=rej.isoformat(),
        grievance_filed_date=gri.isoformat() if gri else None,
        insurer_reply_date=rep.isoformat() if rep else None,
        ombudsman_earliest_date=earliest.isoformat() if earliest else None,
        ombudsman_latest_date=latest.isoformat(), basis=basis, conservative=conservative,
        rules=list(RULES), source=SOURCE, verified=True, verify_note=VERIFY_NOTE)
