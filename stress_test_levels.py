import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from AgentImage import Agent as BaseAgent
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


LEVEL_BUILDERS = [
    build_level_maze_turns,
    build_level_optional_coin,
    build_level_required_multiple,
    build_level_key_door,
    build_level_hazard_detour,
    build_level_pushable_box,
    build_level_power_shield,
    build_level_power_ghost,
    build_level_power_boots,
    build_level_combined_mechanics,
    build_level_boss,
]


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _parse_csv_strs(text: str) -> List[str]:
    out: List[str] = []
    for part in str(text).split(","):
        p = part.strip().lower()
        if p:
            out.append(p)
    return out


def _perturb_image(arr: np.ndarray, mode: str, strength: float, rng: np.random.Generator) -> np.ndarray:
    if mode == "none":
        return arr
    x = arr.astype(np.float32)
    if mode == "brightness":
        factor = float(rng.uniform(1.0 - strength, 1.0 + strength))
        x = x * factor
    elif mode == "contrast":
        factor = float(rng.uniform(1.0 - strength, 1.0 + strength))
        mean = np.mean(x, axis=(0, 1), keepdims=True)
        x = (x - mean) * factor + mean
    elif mode == "noise":
        sigma = 255.0 * max(0.01, strength) * 0.25
        x = x + rng.normal(0.0, sigma, size=x.shape)
    elif mode == "combined":
        b = float(rng.uniform(1.0 - strength, 1.0 + strength))
        c = float(rng.uniform(1.0 - strength, 1.0 + strength))
        mean = np.mean(x, axis=(0, 1), keepdims=True)
        x = ((x * b) - mean) * c + mean
        sigma = 255.0 * max(0.01, strength) * 0.20
        x = x + rng.normal(0.0, sigma, size=x.shape)
    else:
        return arr
    x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


class StressAgent(BaseAgent):
    _use_ml = True
    _perturb_mode = "none"
    _perturb_strength = 0.20
    _seed = 0

    def __init__(self):
        os.environ["AGENT_USE_ML"] = "1" if self._use_ml else "0"
        super().__init__()
        self._step_idx = 0
        mode_id = {"none": 0, "brightness": 1, "contrast": 2, "noise": 3, "combined": 4}.get(
            str(self._perturb_mode), 9
        )
        self._rng = np.random.default_rng(
            int(self._seed) * 1000003 + (1 if self._use_ml else 0) * 10007 + mode_id * 1009
        )

    def step(self, state):
        mode = str(self._perturb_mode)
        if mode == "none":
            return super().step(state)

        if isinstance(state, GridState):
            return super().step(state)

        img = None
        info = {}
        if isinstance(state, dict):
            img = state.get("image")
            maybe_info = state.get("info")
            if isinstance(maybe_info, dict):
                info = maybe_info
        else:
            img = getattr(state, "image", None)
            maybe_info = getattr(state, "info", {})
            if isinstance(maybe_info, dict):
                info = maybe_info

        if img is None:
            return super().step(state)
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return super().step(state)

        self._step_idx += 1
        pert = _perturb_image(arr, mode, float(self._perturb_strength), self._rng)
        wrapped = {"image": pert, "info": info}
        return super().step(wrapped)


@dataclass
class EvalRecord:
    level_idx: int
    level_name: str
    seed: int
    mode: str
    perturb: str
    reward: float
    win: bool
    lose: bool


def _aggregate(records: List[EvalRecord]) -> Dict[Tuple[str, str], Dict[str, float]]:
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in records:
        k = (r.mode, r.perturb)
        cur = out.setdefault(k, {"n": 0.0, "wins": 0.0, "reward_sum": 0.0})
        cur["n"] += 1.0
        cur["wins"] += 1.0 if r.win else 0.0
        cur["reward_sum"] += float(r.reward)
    for v in out.values():
        n = max(1.0, v["n"])
        v["win_rate"] = v["wins"] / n
        v["avg_reward"] = v["reward_sum"] / n
    return out


def _aggregate_by_level(records: List[EvalRecord]) -> Dict[Tuple[str, str, int, str], Dict[str, float]]:
    out: Dict[Tuple[str, str, int, str], Dict[str, float]] = {}
    for r in records:
        k = (r.mode, r.perturb, r.level_idx, r.level_name)
        cur = out.setdefault(k, {"n": 0.0, "wins": 0.0, "reward_sum": 0.0})
        cur["n"] += 1.0
        cur["wins"] += 1.0 if r.win else 0.0
        cur["reward_sum"] += float(r.reward)
    for v in out.values():
        n = max(1.0, v["n"])
        v["win_rate"] = v["wins"] / n
        v["avg_reward"] = v["reward_sum"] / n
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test AgentImage on levels with multi-seed and perturbation sweeps.\n"
            "Example:\n"
            "  python stress_test_levels.py --seeds 10,11,12,13,14 --ml_mode both "
            "--perturbations none,brightness,contrast,noise,combined --strength 0.2"
        )
    )
    parser.add_argument("--seeds", default="10,11,12,13,14")
    parser.add_argument("--ml_mode", default="both", choices=["template", "hybrid", "both"])
    parser.add_argument("--perturbations", default="none")
    parser.add_argument("--strength", type=float, default=0.20)
    parser.add_argument("--levels_from", type=int, default=3)
    parser.add_argument("--levels_to", type=int, default=11)
    parser.add_argument("--turn_limit", type=int, default=150)
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    seeds = _parse_csv_ints(args.seeds)
    if not seeds:
        raise RuntimeError("No valid seeds provided.")

    perturbations = _parse_csv_strs(args.perturbations)
    allowed_perturb = {"none", "brightness", "contrast", "noise", "combined"}
    for p in perturbations:
        if p not in allowed_perturb:
            raise RuntimeError(f"Unsupported perturbation '{p}'. Allowed: {sorted(allowed_perturb)}")

    if args.ml_mode == "both":
        modes = [("template", False), ("hybrid", True)]
    elif args.ml_mode == "template":
        modes = [("template", False)]
    else:
        modes = [("hybrid", True)]

    lo = max(1, int(args.levels_from))
    hi = min(len(LEVEL_BUILDERS), int(args.levels_to))
    if lo > hi:
        raise RuntimeError("levels_from must be <= levels_to")

    selected = []
    for i in range(lo, hi + 1):
        selected.append((i, LEVEL_BUILDERS[i - 1]))

    total_runs = len(seeds) * len(selected) * len(modes) * len(perturbations)
    print(
        f"Running {total_runs} episodes "
        f"(levels={len(selected)}, seeds={len(seeds)}, modes={len(modes)}, perturbations={len(perturbations)})"
    )

    records: List[EvalRecord] = []
    run_idx = 0
    for mode_name, use_ml in modes:
        for perturb in perturbations:
            for seed in seeds:
                for level_idx, builder in selected:
                    run_idx += 1
                    StressAgent._use_ml = bool(use_ml)
                    StressAgent._perturb_mode = str(perturb)
                    StressAgent._perturb_strength = float(args.strength)
                    StressAgent._seed = int(seed)

                    env = create_env(builder, observation_type="image", seed=int(seed), turn_limit=int(args.turn_limit))
                    reward, win, lose, _ = evaluate(StressAgent, env)
                    records.append(
                        EvalRecord(
                            level_idx=level_idx,
                            level_name=builder.__name__,
                            seed=int(seed),
                            mode=mode_name,
                            perturb=str(perturb),
                            reward=float(reward),
                            win=bool(win),
                            lose=bool(lose),
                        )
                    )
                    print(
                        f"[{run_idx:04d}/{total_runs:04d}] "
                        f"{mode_name:8s} perturb={perturb:10s} seed={seed:3d} "
                        f"L{level_idx:02d} {builder.__name__}: reward={reward:.1f} win={win}"
                    )

    overall = _aggregate(records)
    per_level = _aggregate_by_level(records)

    print("\n=== Overall Summary ===")
    for (mode, perturb), stats in sorted(overall.items()):
        print(
            f"{mode:8s} perturb={perturb:10s} "
            f"n={int(stats['n']):4d} win_rate={100.0 * stats['win_rate']:6.2f}% "
            f"avg_reward={stats['avg_reward']:.2f}"
        )

    print("\n=== Per-Level Summary ===")
    for (mode, perturb, level_idx, level_name), stats in sorted(per_level.items()):
        print(
            f"{mode:8s} perturb={perturb:10s} "
            f"L{level_idx:02d} {level_name:30s} "
            f"win_rate={100.0 * stats['win_rate']:6.2f}% avg_reward={stats['avg_reward']:.2f}"
        )

    if args.out_json:
        payload = {
            "config": {
                "seeds": seeds,
                "ml_mode": args.ml_mode,
                "perturbations": perturbations,
                "strength": float(args.strength),
                "levels_from": lo,
                "levels_to": hi,
                "turn_limit": int(args.turn_limit),
            },
            "overall_summary": {
                f"{mode}|{perturb}": stats for (mode, perturb), stats in sorted(overall.items())
            },
            "per_level_summary": {
                f"{mode}|{perturb}|L{level_idx}|{level_name}": stats
                for (mode, perturb, level_idx, level_name), stats in sorted(per_level.items())
            },
            "records": [r.__dict__ for r in records],
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nSaved report: {args.out_json}")


if __name__ == "__main__":
    main()
