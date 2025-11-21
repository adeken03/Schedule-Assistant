from __future__ import annotations

import datetime
from typing import Callable, Dict

from .engine import ScheduleGenerator
from policy import load_active_policy
from wages import wage_amounts


def generate_schedule_for_week(
    session_factory: Callable,
    week_start_date: datetime.date,
    actor: str,
) -> Dict:
    if week_start_date is None:
        raise ValueError("week_start_date is required.")
    with session_factory() as session:
        policy = load_active_policy(session)
        wages = wage_amounts()
        engine = ScheduleGenerator(session, policy, actor=actor or "system", wage_overrides=wages)
        return engine.generate(week_start_date)
