from AgentImage import Agent
from utils import create_env, evaluate

# Use the exact builder names already validated in run_eval.py.
from grid_adventure.levels.intro import (
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
)


def main() -> None:
    levels = [
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

    # "level 3 and above" from this known ordered list.
    for i, builder in enumerate(levels[2:], start=3):
        env = create_env(builder, observation_type="image", seed=10, turn_limit=150)
        r, w, l, _ = evaluate(Agent, env)
        print(f"Level {i} ({builder.__name__}): reward={r}, win={w}, lose={l}")


if __name__ == "__main__":
    main()
