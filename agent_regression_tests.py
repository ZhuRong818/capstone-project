import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Sequence

from Agent import Agent
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
    "combined_mechanics": build_level_combined_mechanics,
    "boss": build_level_boss,
}

PRESETS: Dict[str, Dict[str, str]] = {
    "smoke": {
        "levels": "required_multiple,key_door,hazard_detour,pushable_box,power_shield,power_ghost,power_boots",
        "observation_types": "gridstate,image",
        "seeds": "10",
        "turn_limit": "120",
    },
    "mechanics": {
        "levels": "required_multiple,key_door,hazard_detour,pushable_box,power_shield,power_ghost,power_boots,combined_mechanics",
        "observation_types": "gridstate,image",
        "seeds": "10,11,12",
        "turn_limit": "150",
    },
    "boss": {
        "levels": "combined_mechanics,boss",
        "observation_types": "gridstate,image",
        "seeds": "42,43,44",
        "turn_limit": "150",
    },
    "full": {
        "levels": ",".join(LEVELS.keys()),
        "observation_types": "gridstate,image",
        "seeds": "10,11,12",
        "turn_limit": "150",
    },
}


@dataclass
class EvalRow:
    level: str
    observation_type: str
    seed: int
    reward: float
    win: bool
    lose: bool
    runtime_sec: float


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


def _selected_builders(names: Sequence[str]) -> List[tuple[str, Callable[[], GridState]]]:
    out: List[tuple[str, Callable[[], GridState]]] = []
    for name in names:
        builder = LEVELS.get(name)
        if builder is None:
            raise ValueError(f"Unknown level '{name}'. Choices: {sorted(LEVELS)}")
        out.append((name, builder))
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
    if args.observation_types is None:
        args.observation_types = config["observation_types"]
    if args.seeds is None:
        args.seeds = config["seeds"]
    if args.turn_limit is None:
        args.turn_limit = int(config["turn_limit"])


def run_case(level_name: str, builder: Callable[[], GridState], observation_type: str, seed: int, turn_limit: int) -> EvalRow:
    env = create_env(builder, observation_type=observation_type, seed=seed, turn_limit=turn_limit)
    t0 = time.time()
    reward, win, lose, _ = evaluate(Agent, env)
    dt = time.time() - t0
    return EvalRow(
        level=level_name,
        observation_type=observation_type,
        seed=seed,
        reward=float(reward),
        win=bool(win),
        lose=bool(lose),
        runtime_sec=round(dt, 3),
    )


def summarize(rows: Sequence[EvalRow]) -> dict:
    out = {
        "episodes": len(rows),
        "wins": 0,
        "losses": 0,
        "avg_reward": 0.0,
        "avg_runtime_sec": 0.0,
        "by_bucket": {},
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

        for key in (
            f"level|{row.level}",
            f"obs|{row.observation_type}",
            f"level_obs|{row.level}|{row.observation_type}",
        ):
            bucket = out["by_bucket"].setdefault(
                key,
                {"episodes": 0, "wins": 0, "avg_reward": 0.0, "avg_runtime_sec": 0.0},
            )
            bucket["episodes"] += 1
            bucket["wins"] += 1 if row.win else 0
            bucket["avg_reward"] += row.reward
            bucket["avg_runtime_sec"] += row.runtime_sec

    out["avg_reward"] = reward_sum / len(rows)
    out["avg_runtime_sec"] = runtime_sum / len(rows)
    for bucket in out["by_bucket"].values():
        n = max(1, int(bucket["episodes"]))
        bucket["avg_reward"] /= n
        bucket["avg_runtime_sec"] /= n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run broader local regression tests for Agent.py."
    )
    parser.add_argument(
        "--preset",
        default="mechanics",
        help=f"Named test bundle: {','.join(PRESETS.keys())}",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help=f"Comma-separated subset of: {','.join(LEVELS.keys())}",
    )
    parser.add_argument(
        "--observation_types",
        default=None,
        help="Comma-separated subset of: gridstate,image",
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--turn_limit", type=int, default=None)
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    _apply_preset(args)

    level_names = _parse_csv_strs(args.levels)
    observation_types = _parse_csv_strs(args.observation_types)
    seeds = _parse_csv_ints(args.seeds)
    turn_limit = int(args.turn_limit)

    if not level_names:
        raise ValueError("No levels selected.")
    if not observation_types:
        raise ValueError("No observation types selected.")
    if not seeds:
        raise ValueError("No seeds selected.")
    for obs in observation_types:
        if obs not in ("gridstate", "image"):
            raise ValueError(f"Unsupported observation type '{obs}'.")

    builders = _selected_builders(level_names)
    rows: List[EvalRow] = []
    total = len(builders) * len(observation_types) * len(seeds)
    idx = 0

    for level_name, builder in builders:
        for observation_type in observation_types:
            for seed in seeds:
                idx += 1
                row = run_case(level_name, builder, observation_type, seed, turn_limit)
                rows.append(row)
                print(
                    f"[{idx:02d}/{total:02d}] "
                    f"{level_name:18s} {observation_type:9s} seed={seed:3d} "
                    f"reward={row.reward:7.1f} win={row.win!s:5s} "
                    f"lose={row.lose!s:5s} runtime={row.runtime_sec:6.3f}s"
                )

    summary = summarize(rows)
    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        payload = {
            "preset": args.preset,
            "rows": [asdict(r) for r in rows],
            "summary": summary,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
