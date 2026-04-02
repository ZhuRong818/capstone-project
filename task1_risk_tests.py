import argparse
import importlib.util
import json
import pathlib
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from custom_task1_levels import (
    build_level_boss_shifted_coin,
    build_level_box_two_stage,
    build_level_fallback_ping_pong,
    build_level_lava_commitment_trap,
    build_level_optional_coin_pocket,
    build_level_unreachable_exit_trap,
)
from grid_adventure.grid import GridState
from utils import create_env, evaluate


@dataclass(frozen=True)
class CaseSpec:
    name: str
    issue: str
    description: str
    builder: Callable[[], GridState]
    turn_limit: int


@dataclass
class CaseResult:
    issue: str
    case: str
    seed: int
    reward: float
    win: bool
    lose: bool
    steps: int
    unique_positions: int
    ping_pong: int
    stalled: int
    max_revisit: int
    coins_collected: int
    coins_total: int
    gems_remaining: int
    lava_contacts: int
    boxes_moved: bool
    runtime_sec: float


CASES: Dict[str, CaseSpec] = {
    "fallback_ping_pong": CaseSpec(
        name="fallback_ping_pong",
        issue="1_fallback_strength",
        description="No valid targets, so fallback behavior dominates.",
        builder=build_level_fallback_ping_pong,
        turn_limit=18,
    ),
    "box_two_stage": CaseSpec(
        name="box_two_stage",
        issue="2_box_state_search",
        description="Two-stage box puzzle with dynamic box positions.",
        builder=build_level_box_two_stage,
        turn_limit=80,
    ),
    "boss_shifted_coin": CaseSpec(
        name="boss_shifted_coin",
        issue="3_boss_generalization",
        description="Boss-like map with a slightly shifted intro signature.",
        builder=build_level_boss_shifted_coin,
        turn_limit=120,
    ),
    "unreachable_exit_trap": CaseSpec(
        name="unreachable_exit_trap",
        issue="4_memory_stability",
        description="Unreachable exit behind a locked door with no key.",
        builder=build_level_unreachable_exit_trap,
        turn_limit=30,
    ),
    "optional_coin_pocket": CaseSpec(
        name="optional_coin_pocket",
        issue="5_optional_coin_policy",
        description="Small pocket of profitable optional coins near the exit.",
        builder=build_level_optional_coin_pocket,
        turn_limit=50,
    ),
    "lava_commitment_trap": CaseSpec(
        name="lava_commitment_trap",
        issue="6_lava_budget_policy",
        description="Gem shortcut through lava; shield-first is safer.",
        builder=build_level_lava_commitment_trap,
        turn_limit=60,
    ),
}


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    seen = set()
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p[1:]:
            lo, hi = p.split("-", 1)
            start = int(lo.strip())
            end = int(hi.strip())
            step = 1 if end >= start else -1
            for value in range(start, end + step, step):
                if value not in seen:
                    out.append(value)
                    seen.add(value)
            continue
        value = int(p)
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


def _appearance_name(obj) -> str:
    return str(getattr(getattr(obj, "appearance", None), "name", "")).lower()


def _find_agent_pos(grid_state: GridState) -> Optional[Tuple[int, int]]:
    try:
        width = int(getattr(grid_state, "width", 0))
        height = int(getattr(grid_state, "height", 0))
    except Exception:
        return None
    for x in range(width):
        for y in range(height):
            for obj in grid_state.grid[x][y]:
                name = _appearance_name(obj)
                if getattr(obj, "agent", None) is not None or name in ("agent", "human"):
                    return (x, y)
    return None


def _collect_positions(grid_state: GridState, want: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    try:
        width = int(getattr(grid_state, "width", 0))
        height = int(getattr(grid_state, "height", 0))
    except Exception:
        return out
    for x in range(width):
        for y in range(height):
            for obj in grid_state.grid[x][y]:
                if _appearance_name(obj) == want:
                    out.append((x, y))
    return out


def _count_entities(grid_state: GridState, names: Sequence[str]) -> int:
    want = set(names)
    total = 0
    try:
        width = int(getattr(grid_state, "width", 0))
        height = int(getattr(grid_state, "height", 0))
    except Exception:
        return 0
    for x in range(width):
        for y in range(height):
            for obj in grid_state.grid[x][y]:
                if _appearance_name(obj) in want:
                    total += 1
    return total


def _analyze_history(history: List[Tuple[GridState, object]]) -> Dict[str, int | bool]:
    if not history:
        return {
            "steps": 0,
            "unique_positions": 0,
            "ping_pong": 0,
            "stalled": 0,
            "max_revisit": 0,
            "coins_collected": 0,
            "coins_total": 0,
            "gems_remaining": 0,
            "lava_contacts": 0,
            "boxes_moved": False,
        }

    states = [state for state, _ in history]
    positions = [_find_agent_pos(state) for state in states]
    valid_positions = [pos for pos in positions if pos is not None]
    counts = Counter(valid_positions)

    ping_pong = 0
    stalled = 0
    for i in range(1, len(positions)):
        if positions[i] == positions[i - 1]:
            stalled += 1
    for i in range(2, len(positions)):
        if positions[i] is None or positions[i - 1] is None:
            continue
        if positions[i] == positions[i - 2] and positions[i] != positions[i - 1]:
            ping_pong += 1

    first_state = states[0]
    last_state = states[-1]
    lava_contacts = 0
    for state, pos in zip(states, positions):
        if pos is None:
            continue
        if pos in set(_collect_positions(state, "lava")):
            lava_contacts += 1

    initial_boxes = sorted(_collect_positions(first_state, "box"))
    final_boxes = sorted(_collect_positions(last_state, "box"))

    coins_total = _count_entities(first_state, ("coin",))
    coins_left = _count_entities(last_state, ("coin",))
    gems_remaining = _count_entities(last_state, ("gem", "core"))

    return {
        "steps": max(0, len(history) - 1),
        "unique_positions": len(set(valid_positions)),
        "ping_pong": ping_pong,
        "stalled": stalled,
        "max_revisit": max(counts.values()) if counts else 0,
        "coins_collected": max(0, coins_total - coins_left),
        "coins_total": coins_total,
        "gems_remaining": gems_remaining,
        "lava_contacts": lava_contacts,
        "boxes_moved": initial_boxes != final_boxes,
    }


def _load_agent_class(agent_path: str):
    path = pathlib.Path(agent_path).resolve()
    spec = importlib.util.spec_from_file_location("task1_agent_under_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, "Agent", None)
    if not isinstance(agent_class, type):
        raise AttributeError(f"{path} does not expose an Agent class")
    return agent_class


def _selected_cases(names: Sequence[str]) -> List[CaseSpec]:
    out: List[CaseSpec] = []
    for name in names:
        spec = CASES.get(name)
        if spec is None:
            raise ValueError(f"Unknown case '{name}'. Choices: {sorted(CASES)}")
        out.append(spec)
    return out


def run_case(agent_class, spec: CaseSpec, seed: int) -> CaseResult:
    env = create_env(spec.builder, observation_type="gridstate", seed=seed, turn_limit=spec.turn_limit)
    t0 = time.time()
    reward, win, lose, history = evaluate(agent_class, env)
    dt = time.time() - t0
    metrics = _analyze_history(history)
    return CaseResult(
        issue=spec.issue,
        case=spec.name,
        seed=seed,
        reward=float(reward),
        win=bool(win),
        lose=bool(lose),
        steps=int(metrics["steps"]),
        unique_positions=int(metrics["unique_positions"]),
        ping_pong=int(metrics["ping_pong"]),
        stalled=int(metrics["stalled"]),
        max_revisit=int(metrics["max_revisit"]),
        coins_collected=int(metrics["coins_collected"]),
        coins_total=int(metrics["coins_total"]),
        gems_remaining=int(metrics["gems_remaining"]),
        lava_contacts=int(metrics["lava_contacts"]),
        boxes_moved=bool(metrics["boxes_moved"]),
        runtime_sec=round(dt, 3),
    )


def summarize(rows: Sequence[CaseResult]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "episodes": len(rows),
        "wins": 0,
        "losses": 0,
        "avg_reward": 0.0,
        "avg_runtime_sec": 0.0,
        "by_issue": {},
        "by_case": {},
    }
    if not rows:
        return out

    reward_sum = 0.0
    runtime_sum = 0.0
    for row in rows:
        reward_sum += row.reward
        runtime_sum += row.runtime_sec
        out["wins"] += 1 if row.win else 0
        out["losses"] += 1 if row.lose else 0
        for bucket_name, key in (("by_issue", row.issue), ("by_case", row.case)):
            bucket = out[bucket_name].setdefault(
                key,
                {
                    "episodes": 0,
                    "wins": 0,
                    "avg_reward": 0.0,
                    "avg_steps": 0.0,
                    "avg_unique_positions": 0.0,
                    "avg_ping_pong": 0.0,
                    "avg_max_revisit": 0.0,
                    "avg_coins_collected": 0.0,
                    "avg_lava_contacts": 0.0,
                    "avg_runtime_sec": 0.0,
                },
            )
            bucket["episodes"] += 1
            bucket["wins"] += 1 if row.win else 0
            bucket["avg_reward"] += row.reward
            bucket["avg_steps"] += row.steps
            bucket["avg_unique_positions"] += row.unique_positions
            bucket["avg_ping_pong"] += row.ping_pong
            bucket["avg_max_revisit"] += row.max_revisit
            bucket["avg_coins_collected"] += row.coins_collected
            bucket["avg_lava_contacts"] += row.lava_contacts
            bucket["avg_runtime_sec"] += row.runtime_sec

    out["avg_reward"] = reward_sum / len(rows)
    out["avg_runtime_sec"] = runtime_sum / len(rows)
    for bucket_group in ("by_issue", "by_case"):
        for bucket in out[bucket_group].values():
            n = max(1, int(bucket["episodes"]))
            for key in (
                "avg_reward",
                "avg_steps",
                "avg_unique_positions",
                "avg_ping_pong",
                "avg_max_revisit",
                "avg_coins_collected",
                "avg_lava_contacts",
                "avg_runtime_sec",
            ):
                bucket[key] /= n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run targeted local Task 1 risk probes for issues 1-6."
    )
    parser.add_argument(
        "--agent_path",
        default="task1_proto_min.py",
        help="Path to the agent file exposing an Agent class.",
    )
    parser.add_argument(
        "--cases",
        default=",".join(CASES.keys()),
        help=f"Comma-separated subset of: {','.join(CASES.keys())}",
    )
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    case_names = _parse_csv_strs(args.cases)
    seeds = _parse_csv_ints(args.seeds)
    if not case_names:
        raise ValueError("No cases selected.")
    if not seeds:
        raise ValueError("No seeds selected.")

    agent_class = _load_agent_class(args.agent_path)
    selected = _selected_cases(case_names)

    print(f"Agent: {pathlib.Path(args.agent_path).resolve()}")
    print("Cases:")
    for spec in selected:
        print(f"  - {spec.name}: {spec.issue} | {spec.description}")

    rows: List[CaseResult] = []
    total = len(selected) * len(seeds)
    idx = 0
    for spec in selected:
        for seed in seeds:
            idx += 1
            row = run_case(agent_class, spec, seed)
            rows.append(row)
            print(
                f"[{idx:02d}/{total:02d}] "
                f"{spec.issue:22s} case={spec.name:22s} seed={seed:3d} "
                f"reward={row.reward:7.1f} win={row.win!s:5s} lose={row.lose!s:5s} "
                f"steps={row.steps:3d} uniq={row.unique_positions:3d} "
                f"pingpong={row.ping_pong:3d} revisit={row.max_revisit:3d} "
                f"coins={row.coins_collected}/{row.coins_total} "
                f"lava={row.lava_contacts:2d} boxes_moved={row.boxes_moved!s:5s} "
                f"runtime={row.runtime_sec:6.3f}s"
            )

    summary = summarize(rows)
    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        payload = {
            "rows": [asdict(r) for r in rows],
            "summary": summary,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
