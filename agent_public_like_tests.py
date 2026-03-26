import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from Agent import Agent
from grid_adventure.env import GridAdventureEnv
from grid_adventure.grid import GridState
from grid_adventure.step import Action
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

LEVEL_TURN_LIMIT = 150
TIME_LIMIT = 10
BOSS_TIME_LIMIT = 35


@dataclass(frozen=True)
class CaseSpec:
    name: str
    level: str
    observation_type: str
    seed: int
    min_performance: float = 0.0
    must_win: bool = True


@dataclass
class CaseResult:
    name: str
    level: str
    observation_type: str
    seed: int
    max_total_reward: int
    min_total_reward: int
    performance: float
    total_reward: float
    win: bool
    lose: bool
    timeout: bool
    error: bool
    runtime_sec: float
    min_required_performance: float
    passed: bool


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
    case_name: str
    level: str
    observation_type: str
    seed: int
    total_reward: float
    win: bool
    lose: bool
    steps: List[TraceStep]


PRESETS: Dict[str, List[CaseSpec]] = {
    "debug_boss": [
        CaseSpec(
            name="debug_boss_gridstate_seed1",
            level="boss",
            observation_type="gridstate",
            seed=1,
            min_performance=1.0,
        ),
        CaseSpec(
            name="debug_boss_image_seed2",
            level="boss",
            observation_type="image",
            seed=2,
            min_performance=1.0,
        ),
    ],
    "boss_guard": [
        CaseSpec(name=f"boss_gridstate_seed{seed}", level="boss", observation_type="gridstate", seed=seed)
        for seed in (1, 2, 42, 43, 44)
    ]
    + [
        CaseSpec(name=f"boss_image_seed{seed}", level="boss", observation_type="image", seed=seed)
        for seed in (1, 2, 42, 43, 44)
    ],
    "coursemology_guard": [
        CaseSpec(name=f"combined_gridstate_seed{seed}", level="combined_mechanics", observation_type="gridstate", seed=seed)
        for seed in (10, 11, 12)
    ]
    + [
        CaseSpec(name=f"combined_image_seed{seed}", level="combined_mechanics", observation_type="image", seed=seed)
        for seed in (10, 11, 12)
    ]
    + [
        CaseSpec(name=f"boss_gridstate_seed{seed}", level="boss", observation_type="gridstate", seed=seed)
        for seed in (1, 2, 42, 43, 44)
    ]
    + [
        CaseSpec(name=f"boss_image_seed{seed}", level="boss", observation_type="image", seed=seed)
        for seed in (1, 2, 42, 43, 44)
    ],
    "public_full": [
        CaseSpec(name=f"{level}_gridstate_seed{seed}", level=level, observation_type="gridstate", seed=seed)
        for level in LEVELS.keys()
        for seed in (1, 2, 10, 11, 12, 42, 43, 44)
    ]
    + [
        CaseSpec(name=f"{level}_image_seed{seed}", level=level, observation_type="image", seed=seed)
        for level in LEVELS.keys()
        for seed in (1, 2, 10, 11, 12, 42, 43, 44)
    ],
    "boss_diagnostics": [
        CaseSpec(
            name=f"boss_gridstate_strict_seed{seed}",
            level="boss",
            observation_type="gridstate",
            seed=seed,
            min_performance=1.0,
        )
        for seed in (1, 2, 42, 43, 44)
    ]
    + [
        CaseSpec(
            name=f"boss_image_strict_seed{seed}",
            level="boss",
            observation_type="image",
            seed=seed,
            min_performance=1.0,
        )
        for seed in (1, 2, 42, 43, 44)
    ]
    + [
        CaseSpec(name=f"combined_gridstate_seed{seed}", level="combined_mechanics", observation_type="gridstate", seed=seed)
        for seed in (10, 11, 12)
    ]
    + [
        CaseSpec(name=f"combined_image_seed{seed}", level="combined_mechanics", observation_type="image", seed=seed)
        for seed in (10, 11, 12)
    ],
}


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        p = part.strip()
        if p:
            out.append(int(p))
    return out


def _parse_csv_strs(text: str) -> List[str]:
    out: List[str] = []
    for part in str(text).split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _time_limit_for_level(level: str) -> int:
    return BOSS_TIME_LIMIT if level == "boss" else TIME_LIMIT


def _case_from_axes(
    levels: Sequence[str],
    observation_types: Sequence[str],
    seeds: Sequence[int],
    min_performance: float,
) -> List[CaseSpec]:
    cases: List[CaseSpec] = []
    for level in levels:
        if level not in LEVELS:
            raise ValueError(f"Unknown level '{level}'. Choices: {sorted(LEVELS)}")
        for observation_type in observation_types:
            if observation_type not in ("gridstate", "image"):
                raise ValueError(f"Unsupported observation type '{observation_type}'.")
            for seed in seeds:
                cases.append(
                    CaseSpec(
                        name=f"{level}_{observation_type}_seed{seed}",
                        level=level,
                        observation_type=observation_type,
                        seed=seed,
                        min_performance=min_performance,
                    )
                )
    return cases


def _load_cases(args: argparse.Namespace) -> List[CaseSpec]:
    preset = str(args.preset or "").strip().lower()
    if preset:
        cases = PRESETS.get(preset)
        if cases is None:
            raise ValueError(f"Unknown preset '{preset}'. Choices: {sorted(PRESETS)}")
        return list(cases)

    levels = _parse_csv_strs(args.levels)
    observation_types = _parse_csv_strs(args.observation_types)
    seeds = _parse_csv_ints(args.seeds)
    if not levels:
        raise ValueError("No levels selected.")
    if not observation_types:
        raise ValueError("No observation types selected.")
    if not seeds:
        raise ValueError("No seeds selected.")
    return _case_from_axes(levels, observation_types, seeds, args.min_performance)


def run_case(case: CaseSpec, turn_limit: int) -> CaseResult:
    builder = LEVELS[case.level]
    builder_name = builder.__name__
    max_total_reward = LEVEL_MAX_REWARD[builder_name]
    min_total_reward = LEVEL_MIN_REWARD[builder_name]
    result = evaluate_level(
        Agent,
        builder,
        observation_type=case.observation_type,
        max_total_reward=max_total_reward,
        min_total_reward=min_total_reward,
        turn_limit=turn_limit,
        time_limit=_time_limit_for_level(case.level),
        seed=case.seed,
    )
    passed = True
    if case.must_win and not bool(result["win"]):
        passed = False
    if bool(result["lose"]) or bool(result["timeout"]) or bool(result["error"]):
        passed = False
    if float(result["performance"]) < float(case.min_performance):
        passed = False
    return CaseResult(
        name=case.name,
        level=case.level,
        observation_type=case.observation_type,
        seed=case.seed,
        max_total_reward=max_total_reward,
        min_total_reward=min_total_reward,
        performance=float(result["performance"]),
        total_reward=float(result["total_reward"]),
        win=bool(result["win"]),
        lose=bool(result["lose"]),
        timeout=bool(result["timeout"]),
        error=bool(result["error"]),
        runtime_sec=float(result["runtime (sec)"]),
        min_required_performance=float(case.min_performance),
        passed=passed,
    )


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
    counts = {
        "gems": 0,
        "coins": 0,
        "keys": 0,
        "shields": 0,
        "ghosts": 0,
        "boots": 0,
    }
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


def _trace_case(case: CaseSpec, turn_limit: int) -> CaseTrace:
    builder = LEVELS[case.level]
    env: GridAdventureEnv = create_env(
        builder,
        observation_type=case.observation_type,
        seed=case.seed,
        turn_limit=turn_limit,
    )
    total_reward, win, lose, history = evaluate(Agent, env)
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
        case_name=case.name,
        level=case.level,
        observation_type=case.observation_type,
        seed=case.seed,
        total_reward=float(total_reward),
        win=bool(win),
        lose=bool(lose),
        steps=steps,
    )


def summarize(results: Sequence[CaseResult]) -> dict:
    out = {
        "cases": len(results),
        "passed": 0,
        "failed": 0,
        "wins": 0,
        "avg_reward": 0.0,
        "avg_performance": 0.0,
        "by_bucket": {},
    }
    if not results:
        return out

    reward_sum = 0.0
    perf_sum = 0.0
    for row in results:
        reward_sum += row.total_reward
        perf_sum += row.performance
        out["passed"] += 1 if row.passed else 0
        out["failed"] += 0 if row.passed else 1
        out["wins"] += 1 if row.win else 0
        key = f"{row.level}|{row.observation_type}"
        bucket = out["by_bucket"].setdefault(
            key,
            {
                "cases": 0,
                "passed": 0,
                "wins": 0,
                "avg_reward": 0.0,
                "avg_performance": 0.0,
                "min_reward": None,
                "min_performance": None,
            },
        )
        bucket["cases"] += 1
        bucket["passed"] += 1 if row.passed else 0
        bucket["wins"] += 1 if row.win else 0
        bucket["avg_reward"] += row.total_reward
        bucket["avg_performance"] += row.performance
        bucket["min_reward"] = row.total_reward if bucket["min_reward"] is None else min(bucket["min_reward"], row.total_reward)
        bucket["min_performance"] = (
            row.performance if bucket["min_performance"] is None else min(bucket["min_performance"], row.performance)
        )

    out["avg_reward"] = reward_sum / len(results)
    out["avg_performance"] = perf_sum / len(results)
    for bucket in out["by_bucket"].values():
        n = max(1, int(bucket["cases"]))
        bucket["avg_reward"] /= n
        bucket["avg_performance"] /= n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Coursemology-like local checks with reward/performance thresholds."
    )
    parser.add_argument(
        "--preset",
        default="",
        help=f"Named case bundle: {','.join(PRESETS.keys())}",
    )
    parser.add_argument(
        "--levels",
        default="",
        help=f"Comma-separated subset of: {','.join(LEVELS.keys())}",
    )
    parser.add_argument(
        "--observation_types",
        default="gridstate,image",
        help="Comma-separated subset of: gridstate,image",
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--turn_limit", type=int, default=LEVEL_TURN_LIMIT)
    parser.add_argument("--min_performance", type=float, default=0.0)
    parser.add_argument("--json_out", default="")
    parser.add_argument(
        "--trace_failures_out",
        default="",
        help="Write per-step traces for failed cases to this JSON file.",
    )
    parser.add_argument(
        "--trace_all_out",
        default="",
        help="Write per-step traces for every case to this JSON file.",
    )
    args = parser.parse_args()

    cases = _load_cases(args)
    results: List[CaseResult] = []
    traces: List[CaseTrace] = []
    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        row = run_case(case, args.turn_limit)
        results.append(row)
        status = "PASS" if row.passed else "FAIL"
        print(
            f"[{idx:02d}/{total:02d}] {status} "
            f"{row.name:28s} reward={row.total_reward:7.1f} "
            f"perf={row.performance:5.3f} req={row.min_required_performance:5.3f} "
            f"win={row.win!s:5s} timeout={row.timeout!s:5s} "
            f"error={row.error!s:5s} runtime={row.runtime_sec:6.2f}s"
        )
        should_trace = bool(args.trace_all_out) or (bool(args.trace_failures_out) and not row.passed)
        if should_trace:
            traces.append(_trace_case(case, args.turn_limit))

    summary = summarize(results)
    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        payload = {
            "preset": args.preset,
            "results": [asdict(r) for r in results],
            "summary": summary,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_out}")

    trace_path = args.trace_all_out or args.trace_failures_out
    if trace_path:
        payload = {
            "preset": args.preset,
            "traces": [asdict(t) for t in traces],
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"Wrote {trace_path}")

    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
