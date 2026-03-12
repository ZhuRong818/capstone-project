from AgentImage import Agent
from utils import create_env, evaluate
import os
from grid_adventure.levels.intro import (
    build_level_maze_turns, build_level_optional_coin, build_level_required_multiple,
    build_level_key_door, build_level_hazard_detour, build_level_pushable_box,
    build_level_power_shield, build_level_power_ghost, build_level_power_boots,
    build_level_combined_mechanics, build_level_boss,
)

builders = [
    build_level_maze_turns, build_level_optional_coin, build_level_required_multiple,
    build_level_key_door, build_level_hazard_detour, build_level_pushable_box,
    build_level_power_shield, build_level_power_ghost, build_level_power_boots,
    build_level_combined_mechanics, build_level_boss,
]

run_all = os.environ.get("RUN_ALL_LEVELS", "0") == "1"
selected = builders if run_all else builders[:1]

for i, b in enumerate(selected, 1):
    env = create_env(b, observation_type="image", seed=i, turn_limit=150)
    total_reward, win, lose, _ = evaluate(Agent, env)
    print(f"{b.__name__}: reward={total_reward}, win={win}, lose={lose}")
