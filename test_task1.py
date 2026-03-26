import os
import statistics
import time
import unittest
from dataclasses import dataclass
from typing import Callable, Dict, List

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from Agent import Agent
from grid_adventure.grid import GridState
from grid_adventure.levels.intro import (
    build_level_hazard_detour,
    build_level_key_door,
    build_level_maze_turns,
    build_level_optional_coin,
    build_level_power_boots,
    build_level_power_ghost,
    build_level_power_shield,
    build_level_pushable_box,
    build_level_required_multiple,
)
from utils import create_env, evaluate


LEVELS: Dict[str, Callable[[], GridState]] = {
    "maze_turns": build_level_maze_turns,
    "optional_coin": build_level_optional_coin,
    "required_multiple": build_level_required_multiple,
    "key_door": build_level_key_door,
    "hazard_detour": build_level_hazard_detour,
    "pushable_box": build_level_pushable_box,
    "power_shield": build_level_power_shield,
    "power_ghost": build_level_power_ghost,
    "power_boots": build_level_power_boots,
}

EXPECTED_REWARD: Dict[str, float] = {
    "maze_turns": -27.0,
    "optional_coin": -21.0,
    "required_multiple": -63.0,
    "key_door": -36.0,
    "hazard_detour": -45.0,
    "pushable_box": -21.0,
    "power_shield": -42.0,
    "power_ghost": -48.0,
    "power_boots": -27.0,
}

SEEDS = (10, 11, 12)
TURN_LIMIT = 150
MAX_CASE_RUNTIME_SEC = 0.50
MAX_MEAN_RUNTIME_SEC = 0.25


@dataclass(frozen=True)
class CaseResult:
    level: str
    seed: int
    reward: float
    win: bool
    lose: bool
    runtime_sec: float


def run_case(level: str, seed: int) -> CaseResult:
    env = create_env(LEVELS[level], observation_type="gridstate", seed=seed, turn_limit=TURN_LIMIT)
    started = time.perf_counter()
    reward, win, lose, _ = evaluate(Agent, env)
    runtime_sec = time.perf_counter() - started
    return CaseResult(
        level=level,
        seed=seed,
        reward=float(reward),
        win=bool(win),
        lose=bool(lose),
        runtime_sec=runtime_sec,
    )


class Task1RegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results: List[CaseResult] = []
        for level in LEVELS:
            for seed in SEEDS:
                cls.results.append(run_case(level, seed))

    def test_all_cases_win_with_expected_reward(self) -> None:
        for result in self.results:
            with self.subTest(level=result.level, seed=result.seed):
                self.assertTrue(result.win)
                self.assertFalse(result.lose)
                self.assertEqual(result.reward, EXPECTED_REWARD[result.level])

    def test_task1_runtime_budget(self) -> None:
        runtimes = []
        for result in self.results:
            with self.subTest(level=result.level, seed=result.seed):
                self.assertLess(
                    result.runtime_sec,
                    MAX_CASE_RUNTIME_SEC,
                    f"{result.level} seed={result.seed} took {result.runtime_sec:.3f}s",
                )
                runtimes.append(result.runtime_sec)
        self.assertLess(statistics.mean(runtimes), MAX_MEAN_RUNTIME_SEC)


if __name__ == "__main__":
    unittest.main()
