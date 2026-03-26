import argparse
import importlib.util
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from grid_adventure.env import GridAdventureEnv
from grid_adventure.grid import GridState, from_state, to_state
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
from grid_adventure.rendering import DEFAULT_ASSET_ROOT, IMAGE_MAP
from grid_adventure.step import Action
from utils import get_performance


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

PRESETS: Dict[str, Dict[str, str]] = {
    "smoke": {
        "levels": "hazard_detour,power_ghost,combined_mechanics",
        "seeds": "10",
        "perturbations": "none,combined",
        "ml_modes": "on",
    },
    "public_image": {
        "levels": ",".join(LEVELS.keys()),
        "seeds": "1,2,10,11,12,42,43,44",
        "perturbations": "none",
        "ml_modes": "on",
    },
    "robustness": {
        "levels": "hazard_detour,power_ghost,combined_mechanics,boss",
        "seeds": "1,2,42,43,44",
        "perturbations": "brightness,contrast,noise,combined",
        "ml_modes": "on",
    },
    "compare_ml": {
        "levels": "hazard_detour,power_ghost,combined_mechanics,boss",
        "seeds": "10,11,12",
        "perturbations": "none,combined",
        "ml_modes": "on,off",
    },
    "task2_guard": {
        "levels": ",".join(LEVELS.keys()),
        "seeds": "1,2,10,11,12,42,43,44",
        "perturbations": "none,brightness,contrast,noise,combined",
        "ml_modes": "on",
    },
}

PERTURB_MODES = ("none", "brightness", "contrast", "noise", "combined")
ML_MODES = ("auto", "on", "off")
IMAGE_TIME_LIMIT_SEC = 10.0


@dataclass(frozen=True)
class CaseSpec:
    level: str
    seed: int
    perturb: str
    perturb_strength: float
    ml_mode: str
    asset_profile: str
    render_asset_root: str


@dataclass
class CaseResult:
    level: str
    seed: int
    perturb: str
    perturb_strength: float
    ml_mode: str
    asset_profile: str
    render_asset_root: str
    total_reward: float
    performance: float
    final_score: float
    win: bool
    lose: bool
    timeout: bool
    error: bool
    runtime_sec: float


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
                    seen.add(value)
                    out.append(value)
            continue
        value = int(p)
        if value not in seen:
            seen.add(value)
            out.append(value)
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
    if args.perturbations is None:
        args.perturbations = config["perturbations"]
    if args.ml_modes is None:
        args.ml_modes = config["ml_modes"]


def _load_agent_class(agent_path: str):
    path = Path(agent_path).resolve()
    spec = importlib.util.spec_from_file_location(f"task2_agent_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load agent module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, "Agent", None)
    if agent_class is None:
        raise RuntimeError(f"No Agent class found in {path}")
    return agent_class


def _create_env(
    builder: Callable[[], GridState],
    seed: int,
    turn_limit: int,
    render_asset_root: str,
) -> GridAdventureEnv:
    sample_state = to_state(builder())

    def _initial_state_fn(*args, **kwargs) -> GridState:
        state = to_state(builder())
        state = replace(state, turn_limit=turn_limit)
        return replace(state, seed=seed)

    return GridAdventureEnv(
        initial_state_fn=_initial_state_fn,
        width=sample_state.width,
        height=sample_state.height,
        render_image_map=IMAGE_MAP,
        render_asset_root=render_asset_root,
        observation_type="image",
    )


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
        raise ValueError(f"Unsupported perturbation '{mode}'")
    x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _wrap_observation(state, perturb: str, perturb_strength: float, rng: np.random.Generator):
    if perturb == "none":
        return state

    image = None
    info = {}
    if isinstance(state, dict):
        image = state.get("image")
        maybe_info = state.get("info")
        if isinstance(maybe_info, dict):
            info = maybe_info
    else:
        image = getattr(state, "image", None)
        maybe_info = getattr(state, "info", {})
        if isinstance(maybe_info, dict):
            info = maybe_info

    if image is None:
        return state
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return state
    pert = _perturb_image(arr, perturb, perturb_strength, rng)
    return {"image": pert, "info": info}


def _rng_for_case(case: CaseSpec) -> np.random.Generator:
    perturb_id = {
        "none": 0,
        "brightness": 1,
        "contrast": 2,
        "noise": 3,
        "combined": 4,
    }[case.perturb]
    asset_id = 0 if case.asset_profile == "default" else 1
    ml_id = {"auto": 0, "on": 1, "off": 2}[case.ml_mode]
    seed = int(case.seed) * 1000003 + perturb_id * 1009 + asset_id * 10007 + ml_id * 100003
    return np.random.default_rng(seed)


def run_case(agent_class, case: CaseSpec, threshold: float, turn_limit: int) -> CaseResult:
    builder = LEVELS[case.level]
    env = _create_env(builder, seed=case.seed, turn_limit=turn_limit, render_asset_root=case.render_asset_root)
    builder_name = builder.__name__
    max_total_reward = LEVEL_MAX_REWARD[builder_name]
    min_total_reward = LEVEL_MIN_REWARD[builder_name]

    total_reward = 0.0
    win = False
    lose = False
    error = False
    prev_ml_env = os.environ.get("AGENT_USE_ML")
    if case.ml_mode == "on":
        os.environ["AGENT_USE_ML"] = "1"
    elif case.ml_mode == "off":
        os.environ["AGENT_USE_ML"] = "0"

    rng = _rng_for_case(case)
    started = time.perf_counter()
    try:
        state, _ = env.reset()
        agent = agent_class()
        while not (win or lose):
            observed = _wrap_observation(state, case.perturb, case.perturb_strength, rng)
            action = agent.step(observed)
            state, reward, win, lose, _ = env.step(action)
            total_reward += float(reward)
    except Exception:
        error = True
    runtime_sec = time.perf_counter() - started

    if prev_ml_env is None:
        os.environ.pop("AGENT_USE_ML", None)
    else:
        os.environ["AGENT_USE_ML"] = prev_ml_env

    timeout = runtime_sec >= IMAGE_TIME_LIMIT_SEC
    performance = get_performance(total_reward, max_total_reward, min_total_reward, win)
    if timeout or error or lose or not win:
        performance = 0.0
    final_score = min(1.0, max(0.0, performance / max(1e-9, threshold)))
    return CaseResult(
        level=case.level,
        seed=int(case.seed),
        perturb=case.perturb,
        perturb_strength=float(case.perturb_strength),
        ml_mode=case.ml_mode,
        asset_profile=case.asset_profile,
        render_asset_root=case.render_asset_root,
        total_reward=float(total_reward),
        performance=float(performance),
        final_score=float(final_score),
        win=bool(win),
        lose=bool(lose),
        timeout=bool(timeout),
        error=bool(error),
        runtime_sec=round(runtime_sec, 3),
    )


def summarize(results: Sequence[CaseResult], threshold: float) -> dict:
    out = {
        "cases": len(results),
        "wins": 0,
        "losses": 0,
        "timeouts": 0,
        "errors": 0,
        "avg_reward": 0.0,
        "avg_performance": 0.0,
        "avg_runtime_sec": 0.0,
        "performance_threshold": float(threshold),
        "final_score": 0.0,
        "by_level": {},
        "by_bucket": {},
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

        level_bucket = out["by_level"].setdefault(
            row.level,
            {
                "cases": 0,
                "wins": 0,
                "avg_reward": 0.0,
                "avg_performance": 0.0,
                "avg_runtime_sec": 0.0,
                "min_performance": None,
                "min_reward": None,
            },
        )
        level_bucket["cases"] += 1
        level_bucket["wins"] += 1 if row.win else 0
        level_bucket["avg_reward"] += row.total_reward
        level_bucket["avg_performance"] += row.performance
        level_bucket["avg_runtime_sec"] += row.runtime_sec
        level_bucket["min_performance"] = (
            row.performance
            if level_bucket["min_performance"] is None
            else min(level_bucket["min_performance"], row.performance)
        )
        level_bucket["min_reward"] = (
            row.total_reward
            if level_bucket["min_reward"] is None
            else min(level_bucket["min_reward"], row.total_reward)
        )

        bucket_key = f"{row.asset_profile}|ml={row.ml_mode}|perturb={row.perturb}"
        bucket = out["by_bucket"].setdefault(
            bucket_key,
            {
                "cases": 0,
                "wins": 0,
                "avg_reward": 0.0,
                "avg_performance": 0.0,
                "avg_runtime_sec": 0.0,
                "final_score": 0.0,
            },
        )
        bucket["cases"] += 1
        bucket["wins"] += 1 if row.win else 0
        bucket["avg_reward"] += row.total_reward
        bucket["avg_performance"] += row.performance
        bucket["avg_runtime_sec"] += row.runtime_sec

    out["avg_reward"] = reward_sum / len(results)
    out["avg_performance"] = perf_sum / len(results)
    out["avg_runtime_sec"] = runtime_sum / len(results)
    out["final_score"] = min(1.0, max(0.0, out["avg_performance"] / max(1e-9, threshold)))

    for bucket in out["by_level"].values():
        n = max(1, int(bucket["cases"]))
        bucket["avg_reward"] /= n
        bucket["avg_performance"] /= n
        bucket["avg_runtime_sec"] /= n

    for bucket in out["by_bucket"].values():
        n = max(1, int(bucket["cases"]))
        bucket["avg_reward"] /= n
        bucket["avg_performance"] /= n
        bucket["avg_runtime_sec"] /= n
        bucket["final_score"] = min(1.0, max(0.0, bucket["avg_performance"] / max(1e-9, threshold)))

    out["worst_cases"] = [
        {
            "level": row.level,
            "seed": row.seed,
            "asset_profile": row.asset_profile,
            "ml_mode": row.ml_mode,
            "perturb": row.perturb,
            "performance": row.performance,
            "final_score": row.final_score,
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


def _validate_levels(levels: Sequence[str]) -> None:
    for level in levels:
        if level not in LEVELS:
            raise ValueError(f"Unknown level '{level}'. Choices: {sorted(LEVELS)}")


def _validate_perturbations(perturbations: Sequence[str]) -> None:
    for perturb in perturbations:
        if perturb not in PERTURB_MODES:
            raise ValueError(f"Unknown perturbation '{perturb}'. Choices: {PERTURB_MODES}")


def _validate_ml_modes(modes: Sequence[str]) -> None:
    for mode in modes:
        if mode not in ML_MODES:
            raise ValueError(f"Unknown ml_mode '{mode}'. Choices: {ML_MODES}")


def build_cases(args: argparse.Namespace) -> List[CaseSpec]:
    level_names = _parse_csv_strs(args.levels)
    seeds = _parse_csv_ints(args.seeds)
    perturbations = [p.lower() for p in _parse_csv_strs(args.perturbations)]
    ml_modes = [m.lower() for m in _parse_csv_strs(args.ml_modes)]
    _validate_levels(level_names)
    _validate_perturbations(perturbations)
    _validate_ml_modes(ml_modes)
    if not seeds:
        raise ValueError("No seeds selected.")

    asset_profiles: List[Tuple[str, str]] = [("default", DEFAULT_ASSET_ROOT)]
    external_root = str(args.external_asset_root or "").strip()
    if external_root:
        ext = str(Path(external_root).expanduser().resolve())
        if not Path(ext).is_dir():
            raise ValueError(f"external_asset_root does not exist or is not a directory: {ext}")
        asset_profiles.append(("external", ext))

    cases: List[CaseSpec] = []
    for asset_profile, asset_root in asset_profiles:
        for ml_mode in ml_modes:
            for perturb in perturbations:
                for level in level_names:
                    for seed in seeds:
                        cases.append(
                            CaseSpec(
                                level=level,
                                seed=int(seed),
                                perturb=perturb,
                                perturb_strength=float(args.perturb_strength),
                                ml_mode=ml_mode,
                                asset_profile=asset_profile,
                                render_asset_root=asset_root,
                            )
                        )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensive Task 2 image-observation checks.\n"
            "Covers clean image episodes, synthetic perturbations, optional external sprite roots, "
            "and ML on/off comparisons through the actual submitted agent."
        )
    )
    parser.add_argument("--agent_path", default="Agent.py")
    parser.add_argument("--preset", default="smoke")
    parser.add_argument("--levels", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--perturbations", default=None)
    parser.add_argument("--perturb_strength", type=float, default=0.20)
    parser.add_argument("--ml_modes", default=None)
    parser.add_argument("--turn_limit", type=int, default=150)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--external_asset_root", default="")
    parser.add_argument("--json_out", default="")
    args = parser.parse_args()

    _apply_preset(args)
    cases = build_cases(args)
    agent_class = _load_agent_class(args.agent_path)

    results: List[CaseResult] = []
    total = len(cases)
    for idx, case in enumerate(cases, start=1):
        row = run_case(agent_class, case, threshold=float(args.threshold), turn_limit=int(args.turn_limit))
        results.append(row)
        print(
            f"[{idx:03d}/{total:03d}] "
            f"{case.level:18s} seed={case.seed:3d} "
            f"asset={case.asset_profile:8s} ml={case.ml_mode:4s} "
            f"perturb={case.perturb:10s} "
            f"reward={row.total_reward:7.1f} perf={row.performance:5.3f} "
            f"score={row.final_score:5.3f} win={row.win!s:5s} "
            f"timeout={row.timeout!s:5s} error={row.error!s:5s} "
            f"runtime={row.runtime_sec:6.2f}s"
        )

    summary = summarize(results, threshold=float(args.threshold))
    print("\nSummary:")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.json_out:
        payload = {
            "agent_path": args.agent_path,
            "preset": args.preset,
            "results": [asdict(r) for r in results],
            "summary": summary,
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
