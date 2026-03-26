import argparse
import importlib.util
import json
import pathlib
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from grid_adventure.env import GridAdventureEnv
from grid_adventure.grid import GridState
from grid_adventure.levels.intro import (
    build_level_boss,
    build_level_combined_mechanics,
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
from grid_adventure.step import Action
from utils import create_env, evaluate, evaluate_level


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
    "combined_mechanics": build_level_combined_mechanics,
    "boss": build_level_boss,
}

LEVEL_SCENARIOS: Dict[str, List[str]] = {
    "maze_turns": ["basic_navigation", "maze_turns", "exit_after_requirements"],
    "optional_coin": ["optional_rewards", "reward_vs_progress_pathing"],
    "required_multiple": ["multiple_required_targets", "collection_ordering"],
    "key_door": ["key_pickup", "locked_door", "use_key"],
    "hazard_detour": ["hazard_avoidance", "lava_detour"],
    "pushable_box": ["pushable_box", "movement_feasibility"],
    "power_shield": ["shield_pickup", "hazard_immunity", "lava_crossing"],
    "power_ghost": ["ghost_pickup", "phasing_through_blockers", "locked_or_wall_bypass"],
    "power_boots": ["boots_pickup", "powerup_handling"],
    "combined_mechanics": [
        "combined_mechanics",
        "key_pickup",
        "locked_door",
        "hazard_avoidance",
        "shield_pickup",
        "ghost_pickup",
        "pushable_box",
        "collection_ordering",
    ],
    "boss": [
        "boss_level",
        "large_state_planning",
        "combined_mechanics",
        "resource_management",
    ],
}

LEVEL_MAX_REWARD: Dict[str, int] = {
    build_level_maze_turns.__name__: -27,
    build_level_optional_coin.__name__: -21,
    build_level_required_multiple.__name__: -63,
    build_level_key_door.__name__: -33,
    build_level_hazard_detour.__name__: -39,
    build_level_pushable_box.__name__: -21,
    build_level_power_shield.__name__: -42,
    build_level_power_ghost.__name__: -48,
    build_level_power_boots.__name__: -27,
    build_level_combined_mechanics.__name__: -84,
    build_level_boss.__name__: -68,
}
LEVEL_MIN_REWARD: Dict[str, int] = {k: int(v * 1.5) for k, v in LEVEL_MAX_REWARD.items()}

PRESETS: Dict[str, Dict[str, str]] = {
    "smoke": {
        "levels": "maze_turns,optional_coin,required_multiple,key_door,hazard_detour,pushable_box,power_shield,power_ghost,power_boots",
        "seeds": "10,11,12",
    },
    "task1_hard": {
        "levels": "maze_turns,optional_coin,required_multiple,key_door,hazard_detour,pushable_box,power_shield,power_ghost,power_boots",
        "seeds": "1,2,3,4,5,10,11,12,21,42,43,44",
    },
    "task1_extreme": {
        "levels": "maze_turns,optional_coin,required_multiple,key_door,hazard_detour,pushable_box,power_shield,power_ghost,power_boots",
        "seeds": "1,2,3,4,5,6,7,8,9,10,11,12,13,21,22,23,24,25,42,43,44,45,46,47,48,49,50",
    },
    "task1_weakspots": {
        "levels": "hazard_detour,key_door",
        "seeds": "1,2,3,4,5,6,7,8,9,10,11,12,13,21,22,23,24,25,42,43,44,45,46,47,48,49,50",
    },
    "graded_like": {
        "levels": ",".join(LEVELS.keys()),
        "seeds": "1,2,10,11,12,42,43,44",
    },
    "all_public_25": {
        "levels": ",".join(LEVELS.keys()),
        "seeds": "1-25",
    },
    "all_public_50": {
        "levels": ",".join(LEVELS.keys()),
        "seeds": "1-50",
    },
    "task1_all_scenarios": {
        "levels": "maze_turns,optional_coin,required_multiple,key_door,hazard_detour,pushable_box,power_shield,power_ghost,power_boots",
        "seeds": "1,2,3,4,5,10,11,12,21,22,23,24,25,42,43,44,45,46,47,48,49,50",
    },
}

TURN_LIMIT = 150
TIME_LIMIT = 10
BOSS_TIME_LIMIT = 35


@dataclass(frozen=True)
class CaseSpec:
    level: str
    seed: int


@dataclass
class CaseResult:
    level: str
    seed: int
    total_reward: float
    performance: float
    win: bool
    lose: bool
    timeout: bool
    error: bool
    runtime_sec: float


@dataclass
class TraceStep:
    step: int
    action: str
    reward_before_action: float
    score_after_action: float
    agent_pos_before: Optional[Tuple[int, int]]
    gems_remaining_before: int
    coins_remaining_before: int
    keys_remaining_before: int
    shields_remaining_before: int
    ghosts_remaining_before: int
    boots_remaining_before: int


@dataclass
class CaseTrace:
    level: str
    seed: int
    total_reward: float
    performance: float
    win: bool
    lose: bool
    steps: List[TraceStep]


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue

        range_parts: Optional[Tuple[str, str]] = None
        if ".." in p:
            lo, hi = p.split("..", 1)
            range_parts = (lo.strip(), hi.strip())
        elif "-" in p[1:]:
            lo, hi = p.split("-", 1)
            range_parts = (lo.strip(), hi.strip())

        if range_parts is None:
            value = int(p)
            if value not in seen:
                out.append(value)
                seen.add(value)
            continue

        start = int(range_parts[0])
        end = int(range_parts[1])
        step = 1 if end >= start else -1
        for value in range(start, end + step, step):
            if value not in seen:
                out.append(value)
                seen.add(value)
    return out


def _parse_csv_strs(text: str) -> List[str]:
    out: List[str] = []
    for part in str(text).split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _apply_preset(args: argparse.Namespace) -> None:
    preset = str(args.preset or "").strip().lower()
    if not preset:
        return
    config = PRESETS.get(preset)
    if config is None:
        raise ValueError(f"Unknown preset '{preset}'. Choices: {sorted(PRESETS)}")
    if args.levels is None:
        args.levels = config["levels"]
    if args.seeds is None:
        args.seeds = config["seeds"]


def _time_limit_for_level(level: str) -> int:
    return BOSS_TIME_LIMIT if level == "boss" else TIME_LIMIT


def _load_agent_class(agent_path: str):
    path = pathlib.Path(agent_path).resolve()
    spec = importlib.util.spec_from_file_location(f"task1_agent_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, "Agent", None)
    if agent_class is None:
        raise RuntimeError(f"No Agent class found in {path}")
    return agent_class


def _find_agent_pos(grid_state: GridState) -> Optional[Tuple[int, int]]:
    try:
        width = int(getattr(grid_state, "width", 0))
        height = int(getattr(grid_state, "height", 0))
    except Exception:
        return None
    for x in range(width):
        for y in range(height):
            for obj in grid_state.grid[x][y]:
                name = str(getattr(getattr(obj, "appearance", None), "name", "")).lower()
                if getattr(obj, "agent", None) is not None or name in ("agent", "human"):
                    return (x, y)
    return None


def _count_named_tiles(grid_state: GridState) -> Dict[str, int]:
    counts = {"gems": 0, "coins": 0, "keys": 0, "shields": 0, "ghosts": 0, "boots": 0}
    try:
        width = int(getattr(grid_state, "width", 0))
        height = int(getattr(grid_state, "height", 0))
    except Exception:
        return counts
    for x in range(width):
        for y in range(height):
            for obj in grid_state.grid[x][y]:
                name = str(getattr(getattr(obj, "appearance", None), "name", "")).lower()
                if getattr(obj, "requirable", None) is not None or name in ("gem", "core"):
                    counts["gems"] += 1
                elif getattr(obj, "rewardable", None) is not None and name == "coin":
                    counts["coins"] += 1
                elif getattr(obj, "key", None) is not None or name == "key":
                    counts["keys"] += 1
                elif getattr(obj, "immunity", None) is not None or name == "shield":
                    counts["shields"] += 1
                elif getattr(obj, "phasing", None) is not None or name == "ghost":
                    counts["ghosts"] += 1
                elif getattr(obj, "speed", None) is not None or name == "boots":
                    counts["boots"] += 1
    return counts


def _trace_case(agent_class, case: CaseSpec) -> CaseTrace:
    builder = LEVELS[case.level]
    env: GridAdventureEnv = create_env(builder, observation_type="gridstate", seed=case.seed, turn_limit=TURN_LIMIT)
    total_reward, win, lose, history = evaluate(agent_class, env)
    max_total_reward = LEVEL_MAX_REWARD[builder.__name__]
    min_total_reward = LEVEL_MIN_REWARD[builder.__name__]
    result = evaluate_level(
        agent_class,
        builder,
        observation_type="gridstate",
        max_total_reward=max_total_reward,
        min_total_reward=min_total_reward,
        turn_limit=TURN_LIMIT,
        time_limit=_time_limit_for_level(case.level),
        seed=case.seed,
    )
    steps: List[TraceStep] = []
    running_reward = 0.0
    for idx, (state_before, action) in enumerate(history[:-1], start=1):
        next_state, _ = history[idx]
        counts = _count_named_tiles(state_before)
        score_after = float(getattr(next_state, "score", running_reward) or running_reward)
        steps.append(
            TraceStep(
                step=idx,
                action=action.name if isinstance(action, Action) else str(action),
                reward_before_action=running_reward,
                score_after_action=score_after,
                agent_pos_before=_find_agent_pos(state_before),
                gems_remaining_before=counts["gems"],
                coins_remaining_before=counts["coins"],
                keys_remaining_before=counts["keys"],
                shields_remaining_before=counts["shields"],
                ghosts_remaining_before=counts["ghosts"],
                boots_remaining_before=counts["boots"],
            )
        )
        running_reward = score_after
    return CaseTrace(
        level=case.level,
        seed=case.seed,
        total_reward=float(total_reward),
        performance=float(result["performance"]),
        win=bool(win),
        lose=bool(lose),
        steps=steps,
    )


def run_case(agent_class, case: CaseSpec) -> CaseResult:
    builder = LEVELS[case.level]
    result = evaluate_level(
        agent_class,
        builder,
        observation_type="gridstate",
        max_total_reward=LEVEL_MAX_REWARD[builder.__name__],
        min_total_reward=LEVEL_MIN_REWARD[builder.__name__],
        turn_limit=TURN_LIMIT,
        time_limit=_time_limit_for_level(case.level),
        seed=case.seed,
    )
    return CaseResult(
        level=case.level,
        seed=case.seed,
        total_reward=float(result["total_reward"]),
        performance=float(result["performance"]),
        win=bool(result["win"]),
        lose=bool(result["lose"]),
        timeout=bool(result["timeout"]),
        error=bool(result["error"]),
        runtime_sec=float(result["runtime (sec)"]),
    )


def summarize(results: Sequence[CaseResult]) -> dict:
    covered_levels = sorted({row.level for row in results})
    covered_seeds = sorted({row.seed for row in results})
    covered_scenarios = sorted({scenario for level in covered_levels for scenario in LEVEL_SCENARIOS.get(level, [])})
    out = {
        "cases": len(results),
        "wins": 0,
        "losses": 0,
        "timeouts": 0,
        "errors": 0,
        "avg_reward": 0.0,
        "avg_performance": 0.0,
        "avg_runtime_sec": 0.0,
        "covered_levels": covered_levels,
        "covered_seeds": covered_seeds,
        "seed_count": len(covered_seeds),
        "covered_scenarios": covered_scenarios,
        "by_level": {},
        "by_seed": {},
        "worst_cases": [],
    }
    if not results:
        return out

    reward_sum = 0.0
    perf_sum = 0.0
    runtime_sum = 0.0
    for row in results:
        reward_sum += row.total_reward
        perf_sum += row.performance
        runtime_sum += row.runtime_sec
        out["wins"] += 1 if row.win else 0
        out["losses"] += 1 if row.lose else 0
        out["timeouts"] += 1 if row.timeout else 0
        out["errors"] += 1 if row.error else 0
        bucket = out["by_level"].setdefault(
            row.level,
            {
                "cases": 0,
                "wins": 0,
                "avg_reward": 0.0,
                "avg_performance": 0.0,
                "avg_runtime_sec": 0.0,
                "min_reward": None,
                "min_performance": None,
            },
        )
        bucket["cases"] += 1
        bucket["wins"] += 1 if row.win else 0
        bucket["avg_reward"] += row.total_reward
        bucket["avg_performance"] += row.performance
        bucket["avg_runtime_sec"] += row.runtime_sec
        bucket["min_reward"] = row.total_reward if bucket["min_reward"] is None else min(bucket["min_reward"], row.total_reward)
        bucket["min_performance"] = (
            row.performance if bucket["min_performance"] is None else min(bucket["min_performance"], row.performance)
        )
        seed_bucket = out["by_seed"].setdefault(
            row.seed,
            {
                "cases": 0,
                "wins": 0,
                "avg_reward": 0.0,
                "avg_performance": 0.0,
                "avg_runtime_sec": 0.0,
                "min_reward": None,
                "min_performance": None,
            },
        )
        seed_bucket["cases"] += 1
        seed_bucket["wins"] += 1 if row.win else 0
        seed_bucket["avg_reward"] += row.total_reward
        seed_bucket["avg_performance"] += row.performance
        seed_bucket["avg_runtime_sec"] += row.runtime_sec
        seed_bucket["min_reward"] = (
            row.total_reward if seed_bucket["min_reward"] is None else min(seed_bucket["min_reward"], row.total_reward)
        )
        seed_bucket["min_performance"] = (
            row.performance
            if seed_bucket["min_performance"] is None
            else min(seed_bucket["min_performance"], row.performance)
        )

    out["avg_reward"] = reward_sum / len(results)
    out["avg_performance"] = perf_sum / len(results)
    out["avg_runtime_sec"] = runtime_sum / len(results)
    for bucket in out["by_level"].values():
        n = max(1, int(bucket["cases"]))
        bucket["avg_reward"] /= n
        bucket["avg_performance"] /= n
        bucket["avg_runtime_sec"] /= n
    for bucket in out["by_seed"].values():
        n = max(1, int(bucket["cases"]))
        bucket["avg_reward"] /= n
        bucket["avg_performance"] /= n
        bucket["avg_runtime_sec"] /= n
    out["worst_cases"] = [
        {
            "level": row.level,
            "seed": row.seed,
            "performance": row.performance,
            "total_reward": row.total_reward,
            "win": row.win,
            "lose": row.lose,
            "timeout": row.timeout,
            "error": row.error,
            "runtime_sec": row.runtime_sec,
        }
        for row in sorted(
            results,
            key=lambda r: (
                r.performance,
                0 if (r.error or r.timeout or r.lose or not r.win) else 1,
                r.total_reward,
                -r.runtime_sec,
            ),
        )[:10]
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run comprehensive Task 1 gridstate checks for a target agent file.")
    parser.add_argument("--agent_path", default=".ipynb_checkpoints/task1.py")
    parser.add_argument("--preset", default="graded_like")
    parser.add_argument("--levels", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--json_out", default="")
    parser.add_argument("--trace_failures_out", default="")
    args = parser.parse_args()

    _apply_preset(args)
    level_names = _parse_csv_strs(args.levels)
    seeds = _parse_csv_ints(args.seeds)
    for level in level_names:
        if level not in LEVELS:
            raise ValueError(f"Unknown level '{level}'. Choices: {sorted(LEVELS)}")

    agent_class = _load_agent_class(args.agent_path)
    cases = [CaseSpec(level=level, seed=seed) for level in level_names for seed in seeds]
    results: List[CaseResult] = []
    traces: List[CaseTrace] = []

    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        row = run_case(agent_class, case)
        results.append(row)
        print(
            f"[{idx:02d}/{total:02d}] "
            f"{case.level:18s} seed={case.seed:3d} "
            f"reward={row.total_reward:7.1f} perf={row.performance:5.3f} "
            f"win={row.win!s:5s} lose={row.lose!s:5s} "
            f"timeout={row.timeout!s:5s} error={row.error!s:5s} "
            f"runtime={row.runtime_sec:6.2f}s"
        )
        if args.trace_failures_out and (row.lose or row.timeout or row.error or not row.win):
            traces.append(_trace_case(agent_class, case))

    summary = summarize(results)
    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        payload = {"agent_path": args.agent_path, "results": [asdict(r) for r in results], "summary": summary}
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_out}")

    if args.trace_failures_out:
        with open(args.trace_failures_out, "w", encoding="utf-8") as f:
            json.dump([asdict(t) for t in traces], f, indent=2, sort_keys=True)
        print(f"Wrote {args.trace_failures_out}")


if __name__ == "__main__":
    main()
