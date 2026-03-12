# Input states for Agent step function
from grid_adventure.grid import GridState
from grid_adventure.env import ImageObservation

# State steppers
from grid_adventure.step import Action
from grid_adventure.grid import step as grid_step

# Utility helpers
import math
import os
import random
import copy
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None


class Agent:
    """Task 3 agent: supports both GridState and ImageObservation."""

    def __init__(self):
        self._rng = random.Random(0)
        self._dir_cycle = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        self._dir_idx = 0

        # Perception caches
        self._assets = self._load_assets()
        self._template_cache: Dict[Tuple[int, int], Dict[str, "np.ndarray"]] = {}
        self._tile_cache: Dict[bytes, Optional[str]] = {}
        self._grid_size_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Parse / memory stability
        self._last_agent_pos: Optional[Tuple[int, int]] = None
        self._last_good_state: Optional[GridState] = None
        self._last_good_turn: Optional[int] = None
        self._last_seen_turn: Optional[int] = None
        self._last_flip_y: Optional[bool] = None
        self._consecutive_reuse: int = 0

        # Planning memory
        self._last_step_agent_pos: Optional[Tuple[int, int]] = None
        self._last_step_action: Optional[Action] = None
        self._stuck_steps: int = 0
        self._visit_counts: Dict[Tuple[int, int], int] = {}
        self._expected_pickup_pos: Optional[Tuple[int, int]] = None
        self._expected_pickup_kind: Optional[str] = None

        # Simple mechanic memory
        self._mem_keys: int = 0
        self._mem_shield: bool = False
        self._mem_ghost: bool = False
        self._ghost_turn_duration: int = 5
        self._mem_ghost_turns: int = 0

        # Lightweight parse stats
        self._last_label_counts: Dict[str, int] = {}
        self._step_input_is_image: bool = False

    # -------------------- Public API --------------------

    def step(self, state: GridState | ImageObservation) -> Action:
        self._tick_turn_memory(state)
        self._step_input_is_image = not isinstance(state, GridState)

        grid_state = self._extract_gridstate(state)
        if grid_state is None and self._is_image_observation(state):
            grid_state = self._parse_image_observation(state)

        if grid_state is None:
            act = self._dir_cycle[self._dir_idx]
            self._dir_idx = (self._dir_idx + 1) % len(self._dir_cycle)
            return act

        cur_pos = self._find_agent_pos(grid_state)
        if cur_pos is not None:
            self._visit_counts[cur_pos] = self._visit_counts.get(cur_pos, 0) + 1

        legal = self._legal_actions(grid_state)
        legal_set = set(legal)

        # Emergency: if standing on hazard without shield, prioritize exiting hazard.
        if self._on_hazard_tile(grid_state) and not self._has_shield_active(grid_state):
            for alt in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
                if alt in legal_set and not self._is_lava_move(grid_state, alt):
                    alt = self._anti_stuck_adjust(alt, grid_state, legal)
                    return alt

        if self._should_force_pickup(grid_state) and Action.PICK_UP in legal_set:
            self._apply_expected_pickup_memory()
            self._expected_pickup_pos = None
            self._expected_pickup_kind = None
            return Action.PICK_UP

        act = self._reason_action(grid_state)
        if isinstance(act, Action) and act in legal_set and act != Action.WAIT:
            act = self._anti_stuck_adjust(act, grid_state, legal)
            act = self._avoid_lava_without_shield(act, grid_state, legal)
            return act

        fallback = self._safe_fallback(grid_state, legal)
        fallback = self._anti_stuck_adjust(fallback, grid_state, legal)
        fallback = self._avoid_lava_without_shield(fallback, grid_state, legal)
        return fallback

    # -------------------- Core reasoning --------------------

    def _safe_fallback(
        self, grid_state: Optional[GridState], legal_actions: Optional[List[Action]] = None
    ) -> Action:
        if grid_state is not None:
            legal = legal_actions if legal_actions is not None else self._legal_actions(grid_state)
            if legal:
                if Action.PICK_UP in legal:
                    return Action.PICK_UP
                if Action.USE_KEY in legal:
                    return Action.USE_KEY
                moves = [a for a in legal if a in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)]
                if moves:
                    pos = self._find_agent_pos(grid_state)
                    width = int(getattr(grid_state, "width", 0) or 0)
                    height = int(getattr(grid_state, "height", 0) or 0)
                    if pos is not None:
                        best_act = None
                        best_score = 10**9
                        for act in moves:
                            npos = self._neighbor_pos(pos, act)
                            if npos is None:
                                continue
                            nx, ny = npos
                            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                                continue
                            score = self._visit_counts.get((nx, ny), 0)
                            if score < best_score:
                                best_score = score
                                best_act = act
                        if best_act is not None:
                            return best_act
                for _ in range(len(self._dir_cycle)):
                    act = self._dir_cycle[self._dir_idx]
                    self._dir_idx = (self._dir_idx + 1) % len(self._dir_cycle)
                    if act in legal:
                        return act
                return legal[0]

        act = self._dir_cycle[self._dir_idx]
        self._dir_idx = (self._dir_idx + 1) % len(self._dir_cycle)
        return act

    def _reason_action(self, grid_state: GridState) -> Optional[Action]:
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
            if width <= 0 or height <= 0:
                return None
        except Exception:
            return None

        agent_pos = None
        keys_count = 0
        has_ghost = False
        has_shield = False
        has_boots = False

        gems = set()
        exits = set()
        coins = set()
        keys_on_ground = set()
        shields_on_ground = set()
        ghosts_on_ground = set()
        boots_on_ground = set()
        walls = set()
        locked_doors = set()
        boxes = set()
        lava = set()

        for x in range(width):
            for y in range(height):
                for obj in grid_state.grid[x][y]:
                    n = self._appearance_name(obj)
                    is_agent = getattr(obj, "agent", None) is not None or n in ("agent", "human")
                    if is_agent:
                        if agent_pos is None:
                            agent_pos = (x, y)
                        inv = list(getattr(obj, "inventory_list", None) or [])
                        status = list(getattr(obj, "status_list", None) or [])
                        for item in inv + status:
                            iname = self._appearance_name(item)
                            if getattr(item, "key", None) is not None or iname == "key":
                                keys_count += 1
                            if getattr(item, "immunity", None) is not None or iname == "shield":
                                has_shield = True
                            if getattr(item, "phasing", None) is not None or iname == "ghost":
                                has_ghost = True
                            if getattr(item, "speed", None) is not None or iname == "boots":
                                has_boots = True

                    if getattr(obj, "requirable", None) is not None or n in ("gem", "core"):
                        gems.add((x, y))
                    if getattr(obj, "exit", None) is not None or n == "exit":
                        exits.add((x, y))
                    if getattr(obj, "rewardable", None) is not None and n == "coin":
                        coins.add((x, y))
                    if getattr(obj, "key", None) is not None or n == "key":
                        keys_on_ground.add((x, y))
                    if getattr(obj, "immunity", None) is not None or n == "shield":
                        shields_on_ground.add((x, y))
                    if getattr(obj, "phasing", None) is not None or n == "ghost":
                        ghosts_on_ground.add((x, y))
                    if getattr(obj, "speed", None) is not None or n == "boots":
                        boots_on_ground.add((x, y))
                    if getattr(obj, "locked", None) is not None or n in ("locked", "door_locked"):
                        locked_doors.add((x, y))
                    if getattr(obj, "pushable", None) is not None or n == "box":
                        boxes.add((x, y))
                    is_wall = n == "wall" or (
                        getattr(obj, "blocking", None) is not None
                        and getattr(obj, "locked", None) is None
                        and getattr(obj, "pushable", None) is None
                        and n not in ("door", "opened", "door_open")
                    )
                    if is_wall:
                        walls.add((x, y))
                    if getattr(obj, "damage", None) is not None or n == "lava":
                        lava.add((x, y))

        if agent_pos is None:
            return None

        # Suppress isolated locked-door hallucinations in noisy parses.
        if locked_doors and not (keys_count > 0 or self._mem_keys > 0 or keys_on_ground or exits):
            locked_doors = set()

        keys_count += max(0, int(self._mem_keys))
        has_shield = has_shield or self._mem_shield
        has_ghost = has_ghost or self._mem_ghost or self._mem_ghost_turns > 0
        has_boots = has_boots  # currently not used for blocking, but parsed for completeness
        _ = has_boots

        # Immediate interaction
        if (
            agent_pos in gems
            or agent_pos in coins
            or agent_pos in keys_on_ground
            or agent_pos in shields_on_ground
            or agent_pos in ghosts_on_ground
            or agent_pos in boots_on_ground
        ):
            if agent_pos in keys_on_ground:
                self._mem_keys += 1
            if agent_pos in shields_on_ground:
                self._mem_shield = True
            if agent_pos in ghosts_on_ground:
                self._activate_ghost_memory()
            self._expected_pickup_pos = None
            self._expected_pickup_kind = None
            return Action.PICK_UP

        # Use key if adjacent door
        if keys_count > 0:
            ax, ay = agent_pos
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (ax + dx, ay + dy) in locked_doors:
                    if self._mem_keys > 0:
                        self._mem_keys -= 1
                    return Action.USE_KEY

        door_adj_targets = set()
        if keys_count > 0 and locked_doors:
            for lx, ly in locked_doors:
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    tx, ty = lx + dx, ly + dy
                    if tx < 0 or ty < 0 or tx >= width or ty >= height:
                        continue
                    if (tx, ty) in walls:
                        continue
                    if (tx, ty) in lava and not has_shield:
                        continue
                    door_adj_targets.add((tx, ty))

        need_key_for_door = bool(locked_doors) and keys_count == 0 and bool(keys_on_ground)
        need_shield_for_lava = bool(lava) and not has_shield
        powerups = set(shields_on_ground) | set(ghosts_on_ground) | set(boots_on_ground)

        blocked = set()
        if not has_ghost:
            blocked |= walls
            blocked |= locked_doors
        if not has_shield:
            blocked |= lava
        blocked.discard(agent_pos)

        # If ghost is active, exploit it when phasing opens otherwise unreachable objectives.
        if has_ghost and self._mem_ghost_turns > 0:
            blocked_plain = set(walls) | set(locked_doors)
            if not has_shield:
                blocked_plain |= lava
            blocked_plain.discard(agent_pos)

            blocked_phase = set()
            if not has_shield:
                blocked_phase |= lava
            blocked_phase.discard(agent_pos)

            phase_targets = list(gems) if gems else (list(exits) if exits else list(door_adj_targets))
            if phase_targets:
                plain_action, _ = self._bfs_first_action(agent_pos, phase_targets, width, height, blocked_plain)
                phase_action, phase_target = self._bfs_first_action(agent_pos, phase_targets, width, height, blocked_phase)
                if phase_action is not None and plain_action is None:
                    self._set_expected_pickup(
                        phase_target,
                        keys_on_ground,
                        shields_on_ground,
                        boots_on_ground,
                        ghosts_on_ground,
                    )
                    return phase_action

        # Multi-stage targeting: if primary target is unreachable, try unlockers instead of
        # falling back to random motion.
        target_groups: List[set] = []
        if need_key_for_door and keys_on_ground:
            target_groups.append(set(keys_on_ground))
        if door_adj_targets:
            target_groups.append(set(door_adj_targets))
        if need_shield_for_lava and shields_on_ground:
            target_groups.append(set(shields_on_ground))

        if gems:
            target_groups.append(set(gems))
            if keys_on_ground:
                target_groups.append(set(keys_on_ground))
            if shields_on_ground and not has_shield:
                target_groups.append(set(shields_on_ground))
            if ghosts_on_ground and not has_ghost:
                target_groups.append(set(ghosts_on_ground))
            if boots_on_ground and not has_boots:
                target_groups.append(set(boots_on_ground))
            if powerups:
                target_groups.append(set(powerups))
        else:
            # After required objective, prioritize exit for robustness.
            if exits:
                target_groups.append(set(exits))
            if coins:
                target_groups.append(set(coins))
            if keys_on_ground:
                target_groups.append(set(keys_on_ground))
            if powerups:
                target_groups.append(set(powerups))

        seen_groups = set()
        for group in target_groups:
            if not group:
                continue
            sig = frozenset(group)
            if sig in seen_groups:
                continue
            seen_groups.add(sig)
            action, target = self._bfs_first_action(agent_pos, list(group), width, height, blocked)
            if action is None:
                continue
            if self._is_move_candidate(grid_state, action):
                self._set_expected_pickup(target, keys_on_ground, shields_on_ground, boots_on_ground, ghosts_on_ground)
                return action
            # Avoid repeated impossible first-steps from abstract BFS (often box fronts):
            # block that immediate neighbor once and retry this group.
            bad_step = self._neighbor_pos(agent_pos, action)
            if bad_step is None:
                continue
            blocked_retry = set(blocked)
            blocked_retry.add(bad_step)
            action2, target2 = self._bfs_first_action(agent_pos, list(group), width, height, blocked_retry)
            if action2 is not None and self._is_move_candidate(grid_state, action2):
                self._set_expected_pickup(target2, keys_on_ground, shields_on_ground, boots_on_ground, ghosts_on_ground)
                return action2

        self._expected_pickup_pos = None
        self._expected_pickup_kind = None
        return None

    # -------------------- Move filtering --------------------

    def _appearance_name(self, obj) -> str:
        return str(getattr(getattr(obj, "appearance", None), "name", "")).lower()

    def _find_agent_pos(self, grid_state: GridState) -> Optional[Tuple[int, int]]:
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
        except Exception:
            return None
        for x in range(width):
            for y in range(height):
                for obj in grid_state.grid[x][y]:
                    if getattr(obj, "agent", None) is not None or self._appearance_name(obj) in ("agent", "human"):
                        return (x, y)
        return None

    def _neighbor_pos(self, pos: Tuple[int, int], act: Action) -> Optional[Tuple[int, int]]:
        dxdy = {
            Action.RIGHT: (1, 0),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.UP: (0, -1),
        }
        if act not in dxdy:
            return None
        dx, dy = dxdy[act]
        return (int(pos[0]) + dx, int(pos[1]) + dy)

    def _has_shield_active(self, grid_state: GridState) -> bool:
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return self._mem_shield
        ax, ay = pos
        try:
            for obj in grid_state.grid[ax][ay]:
                is_agent = getattr(obj, "agent", None) is not None or self._appearance_name(obj) in ("agent", "human")
                if not is_agent:
                    continue
                inv = list(getattr(obj, "inventory_list", None) or [])
                status = list(getattr(obj, "status_list", None) or [])
                for item in inv + status:
                    if getattr(item, "immunity", None) is not None or self._appearance_name(item) == "shield":
                        return True
        except Exception:
            pass
        return self._mem_shield

    def _has_ghost_active(self, grid_state: GridState) -> bool:
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return self._mem_ghost or self._mem_ghost_turns > 0
        ax, ay = pos
        try:
            for obj in grid_state.grid[ax][ay]:
                is_agent = getattr(obj, "agent", None) is not None or self._appearance_name(obj) in ("agent", "human")
                if not is_agent:
                    continue
                inv = list(getattr(obj, "inventory_list", None) or [])
                status = list(getattr(obj, "status_list", None) or [])
                for item in inv + status:
                    if getattr(item, "phasing", None) is not None or self._appearance_name(item) == "ghost":
                        return True
        except Exception:
            pass
        return self._mem_ghost or self._mem_ghost_turns > 0

    def _is_lava_move(self, grid_state: GridState, act: Action) -> bool:
        if act not in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            return False
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return False
        nxt = self._neighbor_pos(pos, act)
        if nxt is None:
            return False
        nx, ny = nxt
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                return False
            for obj in grid_state.grid[nx][ny]:
                if getattr(obj, "damage", None) is not None or self._appearance_name(obj) == "lava":
                    return True
        except Exception:
            return False
        return False

    def _avoid_lava_without_shield(self, act: Action, grid_state: GridState, legal: List[Action]) -> Action:
        if self._has_shield_active(grid_state):
            return act
        if not self._is_lava_move(grid_state, act):
            return act
        for alt in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            if alt in legal and not self._is_lava_move(grid_state, alt):
                return alt
        return act

    def _on_collectible_tile(self, grid_state: GridState) -> bool:
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return False
        x, y = pos
        try:
            for obj in grid_state.grid[x][y]:
                n = self._appearance_name(obj)
                if getattr(obj, "requirable", None) is not None or n in ("gem", "core"):
                    return True
                if getattr(obj, "rewardable", None) is not None and n == "coin":
                    return True
                if getattr(obj, "key", None) is not None or n == "key":
                    return True
                if getattr(obj, "speed", None) is not None or n == "boots":
                    return True
                if getattr(obj, "phasing", None) is not None or n == "ghost":
                    return True
                if getattr(obj, "immunity", None) is not None or n == "shield":
                    return True
        except Exception:
            return False
        return False

    def _on_hazard_tile(self, grid_state: GridState) -> bool:
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return False
        x, y = pos
        try:
            for obj in grid_state.grid[x][y]:
                n = self._appearance_name(obj)
                if getattr(obj, "damage", None) is not None or n == "lava":
                    return True
        except Exception:
            return False
        return False

    def _should_force_pickup(self, grid_state: GridState) -> bool:
        if self._on_collectible_tile(grid_state):
            return True
        pos = self._find_agent_pos(grid_state)
        return pos is not None and self._expected_pickup_pos is not None and pos == self._expected_pickup_pos

    def _can_use_key_now(self, grid_state: GridState) -> bool:
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return False
        ax, ay = pos
        has_key = False
        try:
            for obj in grid_state.grid[ax][ay]:
                is_agent = getattr(obj, "agent", None) is not None or self._appearance_name(obj) in ("agent", "human")
                if not is_agent:
                    continue
                inv = list(getattr(obj, "inventory_list", None) or [])
                for item in inv:
                    if getattr(item, "key", None) is not None or self._appearance_name(item) == "key":
                        has_key = True
                        break
                if has_key:
                    break
            if not has_key and self._mem_keys > 0:
                has_key = True
            if not has_key:
                return False
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = ax + dx, ay + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                for obj in grid_state.grid[nx][ny]:
                    n = self._appearance_name(obj)
                    if getattr(obj, "locked", None) is not None or n in ("locked", "door_locked"):
                        return True
        except Exception:
            return False
        return False

    def _is_move_candidate(self, grid_state: GridState, act: Action) -> bool:
        if act not in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            return False
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return True
        nxt = self._neighbor_pos(pos, act)
        if nxt is None:
            return False
        nx, ny = nxt
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                return False
            has_ghost = self._has_ghost_active(grid_state)
            dx, dy = nx - pos[0], ny - pos[1]
            for obj in grid_state.grid[nx][ny]:
                n = self._appearance_name(obj)
                is_locked = getattr(obj, "locked", None) is not None or n in ("locked", "door_locked")
                if is_locked and not has_ghost:
                    return False
                is_wall = n == "wall" or (
                    getattr(obj, "blocking", None) is not None
                    and getattr(obj, "locked", None) is None
                    and getattr(obj, "pushable", None) is None
                    and n not in ("door", "opened", "door_open")
                )
                if is_wall and not has_ghost:
                    return False
                is_box = getattr(obj, "pushable", None) is not None or n == "box"
                if is_box and not has_ghost:
                    bx, by = nx + dx, ny + dy
                    if bx < 0 or by < 0 or bx >= width or by >= height:
                        return False
                    for bobj in grid_state.grid[bx][by]:
                        bn = self._appearance_name(bobj)
                        b_locked = getattr(bobj, "locked", None) is not None or bn in ("locked", "door_locked")
                        b_wall = bn == "wall" or (
                            getattr(bobj, "blocking", None) is not None
                            and getattr(bobj, "locked", None) is None
                            and getattr(bobj, "pushable", None) is None
                            and bn not in ("door", "opened", "door_open")
                        )
                        b_box = getattr(bobj, "pushable", None) is not None or bn == "box"
                        if b_locked or b_wall or b_box:
                            return False
            return True
        except Exception:
            return True

    def _legal_actions(self, grid_state: GridState) -> List[Action]:
        if not self._step_input_is_image:
            return self._legal_actions_exact(grid_state)

        legal: List[Action] = []
        if self._should_force_pickup(grid_state):
            legal.append(Action.PICK_UP)
        if self._can_use_key_now(grid_state):
            legal.append(Action.USE_KEY)
        move_candidates: List[Action] = []
        safe_moves: List[Action] = []
        has_shield = self._has_shield_active(grid_state)
        for act in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            if not self._is_move_candidate(grid_state, act):
                continue
            move_candidates.append(act)
            if has_shield or not self._is_lava_move(grid_state, act):
                safe_moves.append(act)
        if safe_moves:
            legal.extend(safe_moves)
        else:
            legal.extend(move_candidates)
        if not legal:
            legal.append(Action.WAIT)
        return legal

    def _legal_actions_exact(self, grid_state: GridState) -> List[Action]:
        legal: List[Action] = []
        for act in (
            Action.PICK_UP,
            Action.USE_KEY,
            Action.RIGHT,
            Action.DOWN,
            Action.LEFT,
            Action.UP,
        ):
            try:
                if act == Action.PICK_UP and not self._should_force_pickup(grid_state):
                    continue
                if act == Action.USE_KEY and not self._can_use_key_now(grid_state):
                    continue
                trial_state = copy.deepcopy(grid_state)
                result = grid_step(trial_state, act)
                next_state = result[0] if isinstance(result, tuple) else result
                if next_state is not None:
                    legal.append(act)
            except Exception:
                continue

        # Prefer non-lava moves when there is a safe alternative.
        if not self._has_shield_active(grid_state):
            moves = [a for a in legal if a in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)]
            safe_moves = [a for a in moves if not self._is_lava_move(grid_state, a)]
            if safe_moves:
                keep = [a for a in legal if a not in moves]
                keep.extend(safe_moves)
                legal = keep

        if not legal:
            legal.append(Action.WAIT)
        return legal

    def _bfs_first_action(
        self,
        start: Tuple[int, int],
        targets: List[Tuple[int, int]],
        width: int,
        height: int,
        blocked: set,
    ) -> Tuple[Optional[Action], Optional[Tuple[int, int]]]:
        target_set = set(targets)
        if start in target_set:
            return Action.PICK_UP, start

        q = deque([start])
        parent: Dict[Tuple[int, int], Tuple[Tuple[int, int], Action]] = {}
        seen = {start}
        dirs = [
            (1, 0, Action.RIGHT),
            (-1, 0, Action.LEFT),
            (0, 1, Action.DOWN),
            (0, -1, Action.UP),
        ]

        found = None
        while q:
            cur = q.popleft()
            if cur in target_set:
                found = cur
                break
            cx, cy = cur
            for dx, dy, act in dirs:
                nx, ny = cx + dx, cy + dy
                nxt = (nx, ny)
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if nxt in blocked or nxt in seen:
                    continue
                seen.add(nxt)
                parent[nxt] = (cur, act)
                q.append(nxt)

        if found is None:
            return None, None
        cur = found
        while parent.get(cur, (None, None))[0] != start:
            prev = parent.get(cur)
            if prev is None:
                return None, None
            cur = prev[0]
        first = parent.get(cur)
        return (first[1] if first is not None else None), found

    def _set_expected_pickup(
        self,
        found_target: Optional[Tuple[int, int]],
        keys_on_ground: set,
        shields_on_ground: set,
        boots_on_ground: set,
        ghosts_on_ground: set,
    ) -> None:
        self._expected_pickup_pos = found_target
        self._expected_pickup_kind = None
        if found_target is None:
            return
        if found_target in keys_on_ground:
            self._expected_pickup_kind = "key"
        elif found_target in shields_on_ground:
            self._expected_pickup_kind = "shield"
        elif found_target in boots_on_ground:
            self._expected_pickup_kind = "boots"
        elif found_target in ghosts_on_ground:
            self._expected_pickup_kind = "ghost"

    def _apply_expected_pickup_memory(self) -> None:
        if self._expected_pickup_kind == "key":
            self._mem_keys += 1
        elif self._expected_pickup_kind == "shield":
            self._mem_shield = True
        elif self._expected_pickup_kind == "ghost":
            self._activate_ghost_memory()

    def _pick_escape_move(self, action: Action, grid_state: GridState, legal: List[Action]) -> Optional[Action]:
        moves = (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)
        alts = [alt for alt in moves if alt in legal and alt != action]
        if not alts:
            return None
        cur = self._find_agent_pos(grid_state)
        if cur is None:
            return alts[0]
        width = int(getattr(grid_state, "width", 0) or 0)
        height = int(getattr(grid_state, "height", 0) or 0)
        best_alt = None
        best_score = None
        for alt in alts:
            npos = self._neighbor_pos(cur, alt)
            if npos is None:
                continue
            nx, ny = npos
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            score = self._visit_counts.get((nx, ny), 0)
            if best_score is None or score < best_score:
                best_score = score
                best_alt = alt
        if best_alt is not None:
            return best_alt
        return alts[0]

    def _anti_stuck_adjust(self, action: Action, grid_state: GridState, legal: List[Action]) -> Action:
        moves = (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)
        pos = self._find_agent_pos(grid_state)
        if (
            pos is not None
            and self._last_step_agent_pos == pos
            and action in moves
            and self._last_step_action == action
        ):
            escape = self._pick_escape_move(action, grid_state, legal)
            if escape is not None:
                self._last_step_action = escape
                self._last_step_agent_pos = pos
                self._stuck_steps = 0
                return escape

        if pos is not None and self._last_step_agent_pos == pos and self._last_step_action == action:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0

        self._last_step_agent_pos = pos
        self._last_step_action = action

        if self._stuck_steps < 3:
            return action

        escape = self._pick_escape_move(action, grid_state, legal)
        if escape is not None:
            self._last_step_action = escape
            self._stuck_steps = 0
            return escape
        return action

    # -------------------- Memory ticking --------------------

    def _tick_turn_memory(self, state: GridState | ImageObservation) -> None:
        turn = None
        try:
            if isinstance(state, GridState):
                maybe_turn = getattr(state, "turn", None)
                turn = int(maybe_turn) if maybe_turn is not None else None
            else:
                _, info = self._extract_observation(state)
                turn = self._extract_turn(info)
        except Exception:
            turn = None

        advanced = False
        if turn is not None:
            if self._last_seen_turn is not None and turn < self._last_seen_turn:
                self._reset_episode_memory()
            if self._last_seen_turn is None or turn > self._last_seen_turn:
                advanced = True
            self._last_seen_turn = turn
        else:
            advanced = True

        if not advanced:
            return

        if self._mem_ghost_turns > 0:
            self._mem_ghost_turns -= 1
        if self._mem_ghost_turns <= 0:
            self._mem_ghost_turns = 0
            self._mem_ghost = False

    def _reset_episode_memory(self) -> None:
        self._last_step_agent_pos = None
        self._last_step_action = None
        self._stuck_steps = 0
        self._visit_counts.clear()
        self._expected_pickup_pos = None
        self._expected_pickup_kind = None
        self._mem_keys = 0
        self._mem_shield = False
        self._mem_ghost = False
        self._mem_ghost_turns = 0
        self._consecutive_reuse = 0

    def _activate_ghost_memory(self) -> None:
        self._mem_ghost = True
        self._mem_ghost_turns = max(self._mem_ghost_turns, self._ghost_turn_duration + 1)

    # -------------------- Observation handling --------------------

    def _is_image_observation(self, state) -> bool:
        if isinstance(state, dict) and "image" in state:
            return True
        return hasattr(state, "image")

    def _extract_observation(self, observation):
        if isinstance(observation, dict):
            return observation.get("image"), observation.get("info", {})
        image = getattr(observation, "image", None)
        info = getattr(observation, "info", {})
        return image, info

    def _extract_gridstate(self, state: GridState | ImageObservation) -> Optional[GridState]:
        if isinstance(state, GridState):
            return state
        if isinstance(state, dict):
            info = state.get("info", {})
            maybe = info.get("gridstate") if isinstance(info, dict) else None
            if isinstance(maybe, GridState):
                return maybe
            return None
        info = getattr(state, "info", {})
        maybe = info.get("gridstate") if isinstance(info, dict) else None
        if isinstance(maybe, GridState):
            return maybe
        return None

    def _extract_turn(self, info) -> Optional[int]:
        if not isinstance(info, dict):
            return None
        turn = info.get("turn")
        if turn is None:
            status = info.get("status")
            if isinstance(status, dict):
                turn = status.get("turn")
        if turn is None:
            return None
        try:
            return int(turn)
        except Exception:
            return None

    def _reusable_last_state(self, info) -> Optional[GridState]:
        if self._last_good_state is None:
            return None
        current_turn = self._extract_turn(info)
        if current_turn is None or self._last_good_turn is None:
            return self._last_good_state
        if 0 <= current_turn - self._last_good_turn <= 1:
            return self._last_good_state
        return None

    def _hinted_agent_pos(
        self, info, grid_w: int, grid_h: int, use_memory: bool = True
    ) -> Optional[Tuple[int, int]]:
        if isinstance(info, dict):
            maybe = info.get("agent_pos") or info.get("player_pos") or info.get("position")
            if isinstance(maybe, (list, tuple)) and len(maybe) == 2:
                try:
                    ax, ay = int(maybe[0]), int(maybe[1])
                    if 0 <= ax < grid_w and 0 <= ay < grid_h:
                        return (ax, ay)
                except Exception:
                    pass
        if use_memory and self._last_agent_pos is not None:
            ax, ay = self._last_agent_pos
            if 0 <= ax < grid_w and 0 <= ay < grid_h:
                return (ax, ay)
        return None

    def _to_grid_pos(self, x: int, y: int, grid_h: int, flip_y: bool) -> Tuple[int, int]:
        if flip_y:
            return (x, grid_h - 1 - y)
        return (x, y)

    def _choose_flip_y(
        self,
        recognized: List[Tuple[int, int, str]],
        hinted_agent: Optional[Tuple[int, int]],
        grid_h: int,
    ) -> bool:
        if hinted_agent is None:
            return False
        agent_tiles = [(x, y) for x, y, label in recognized if label == "agent"]
        if not agent_tiles:
            return False

        hx, hy = hinted_agent

        def nearest_distance(flip_y: bool) -> int:
            best = 10**9
            for ax, ay in agent_tiles:
                gx, gy = self._to_grid_pos(ax, ay, grid_h, flip_y)
                best = min(best, abs(gx - hx) + abs(gy - hy))
            return best

        return nearest_distance(True) < nearest_distance(False)

    # -------------------- Image parsing --------------------

    def _parse_image_observation(self, observation) -> Optional[GridState]:
        if np is None:
            return self._reusable_last_state({})

        image, info = self._extract_observation(observation)
        if image is None:
            return self._reusable_last_state(info)

        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] < 3:
            return self._reusable_last_state(info)

        grid_w, grid_h = self._infer_grid_size(info, img.shape[1], img.shape[0])
        if not grid_w or not grid_h:
            return self._reusable_last_state(info)

        tile_w = img.shape[1] // grid_w
        tile_h = img.shape[0] // grid_h
        if tile_w <= 0 or tile_h <= 0:
            return self._reusable_last_state(info)

        state = self._make_empty_gridstate(grid_w, grid_h)
        if state is None:
            return self._reusable_last_state(info)

        try:
            from grid_adventure.entities import (
                AgentEntity,
                BoxEntity,
                CoinEntity,
                ExitEntity,
                GemEntity,
                KeyEntity,
                LavaEntity,
                LockedDoorEntity,
                PhasingPowerUpEntity,
                ShieldPowerUpEntity,
                SpeedPowerUpEntity,
                WallEntity,
            )
            try:
                from grid_adventure.entities import UnlockedDoorEntity
            except Exception:
                UnlockedDoorEntity = None
        except Exception:
            return self._reusable_last_state(info)

        entity_by_label: Dict[str, type] = {
            "agent": AgentEntity,
            "gem": GemEntity,
            "coin": CoinEntity,
            "key": KeyEntity,
            "boots": SpeedPowerUpEntity,
            "ghost": PhasingPowerUpEntity,
            "shield": ShieldPowerUpEntity,
            "box": BoxEntity,
            "door_locked": LockedDoorEntity,
            "exit": ExitEntity,
            "wall": WallEntity,
            "lava": LavaEntity,
        }
        if UnlockedDoorEntity is not None:
            entity_by_label["door_open"] = UnlockedDoorEntity

        recognized: List[Tuple[int, int, str]] = []
        agent_pos: Optional[Tuple[int, int]] = None
        label_counts: Dict[str, int] = {}

        for y in range(grid_h):
            for x in range(grid_w):
                tile = img[y * tile_h : (y + 1) * tile_h, x * tile_w : (x + 1) * tile_w]
                label = self._classify_tile(tile, tile_h, tile_w)
                if label is None:
                    label = self._heuristic_tile(tile)
                if label is None or label == "floor":
                    continue
                recognized.append((x, y, label))
                label_counts[label] = label_counts.get(label, 0) + 1

        # Recover common confusion: single right-edge door may really be exit.
        if int(label_counts.get("exit", 0)) == 0 and int(label_counts.get("door_locked", 0)) == 1:
            for i, (x, y, label) in enumerate(recognized):
                if label != "door_locked":
                    continue
                if x >= max(0, grid_w - 3) and int(label_counts.get("key", 0)) == 0:
                    recognized[i] = (x, y, "exit")
                    label_counts["door_locked"] = max(0, int(label_counts.get("door_locked", 0)) - 1)
                    label_counts["exit"] = int(label_counts.get("exit", 0)) + 1
                break

        self._last_label_counts = dict(label_counts)

        hinted_agent = self._hinted_agent_pos(info, grid_w, grid_h, use_memory=True)
        flip_y = self._choose_flip_y(recognized, hinted_agent, grid_h)
        self._last_flip_y = flip_y

        agent_raw_tiles = [(x, y) for x, y, label in recognized if label == "agent"]
        chosen_agent_raw = None
        if agent_raw_tiles:
            if hinted_agent is not None:
                hx, hy = hinted_agent
                chosen_agent_raw = min(
                    agent_raw_tiles,
                    key=lambda p: abs(self._to_grid_pos(p[0], p[1], grid_h, flip_y)[0] - hx)
                    + abs(self._to_grid_pos(p[0], p[1], grid_h, flip_y)[1] - hy),
                )
            else:
                chosen_agent_raw = agent_raw_tiles[0]

        for x, y, label in recognized:
            if label == "agent" and chosen_agent_raw is not None and (x, y) != chosen_agent_raw:
                continue
            ent_cls = entity_by_label.get(label)
            if ent_cls is None:
                continue
            game_pos = self._to_grid_pos(x, y, grid_h, flip_y)
            state.add(game_pos, ent_cls())
            if label == "agent":
                agent_pos = game_pos

        if agent_pos is None:
            hinted = self._hinted_agent_pos(info, grid_w, grid_h)
            if hinted is not None:
                agent_pos = hinted
                state.add(agent_pos, entity_by_label["agent"]())

        if agent_pos is not None and self._is_plausible_parse(grid_w, grid_h, label_counts, state):
            self._last_agent_pos = agent_pos
            self._last_good_state = state
            self._last_good_turn = self._extract_turn(info)
            self._consecutive_reuse = 0
            return state

        reused = self._reusable_last_state(info)
        if reused is not None and self._consecutive_reuse < 3:
            self._consecutive_reuse += 1
            return reused
        self._consecutive_reuse = 0
        return None

    def _is_plausible_parse(
        self,
        grid_w: int,
        grid_h: int,
        label_counts: Dict[str, int],
        state: GridState,
    ) -> bool:
        cells = max(1, grid_w * grid_h)
        agent_n = int(label_counts.get("agent", 0))
        if agent_n > 2:
            return False
        if agent_n == 0 and self._find_agent_pos(state) is None:
            return False
        if label_counts.get("door_locked", 0) > max(3, cells // 5):
            return False
        if label_counts.get("box", 0) > max(4, cells // 4):
            return False
        if label_counts.get("shield", 0) + label_counts.get("ghost", 0) + label_counts.get("boots", 0) > max(4, cells // 4):
            return False
        blocked_like = int(label_counts.get("wall", 0)) + int(label_counts.get("lava", 0))
        if blocked_like > int(cells * 0.85):
            return False
        if self._find_agent_pos(state) is None:
            return False
        return True

    # -------------------- Grid size inference --------------------

    def _infer_grid_size(
        self, info, width_px: int, height_px: int
    ) -> Tuple[Optional[int], Optional[int]]:
        if isinstance(info, dict):
            config = info.get("config")
            if isinstance(config, dict):
                for w_key, h_key in (
                    ("width", "height"),
                    ("grid_width", "grid_height"),
                    ("cols", "rows"),
                    ("n_cols", "n_rows"),
                ):
                    if w_key in config and h_key in config:
                        try:
                            w, h = int(config[w_key]), int(config[h_key])
                            if w > 0 and h > 0:
                                return w, h
                        except Exception:
                            pass
            for w_key, h_key in (
                ("width", "height"),
                ("grid_width", "grid_height"),
                ("cols", "rows"),
                ("n_cols", "n_rows"),
            ):
                if w_key in info and h_key in info:
                    try:
                        w, h = int(info[w_key]), int(info[h_key])
                        if w > 0 and h > 0:
                            return w, h
                    except Exception:
                        pass
            shape = info.get("grid_shape") or info.get("shape") or info.get("size")
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                try:
                    w, h = int(shape[0]), int(shape[1])
                    if w > 0 and h > 0:
                        return w, h
                except Exception:
                    pass

        if width_px <= 0 or height_px <= 0:
            return None, None

        key = (width_px, height_px)
        if key in self._grid_size_cache:
            return self._grid_size_cache[key]

        g = math.gcd(width_px, height_px)
        candidates: List[Tuple[int, int, int]] = []
        for d in range(1, int(g**0.5) + 1):
            if g % d != 0:
                continue
            for tile in (d, g // d):
                if tile < 8 or tile > 128:
                    continue
                if width_px % tile != 0 or height_px % tile != 0:
                    continue
                gw, gh = width_px // tile, height_px // tile
                if 1 <= gw <= 24 and 1 <= gh <= 24:
                    candidates.append((gw, gh, tile))

        if not candidates:
            return None, None

        for gw, gh, tile in candidates:
            if tile == 64:
                self._grid_size_cache[key] = (gw, gh)
                return gw, gh

        best = min(candidates, key=lambda t: abs(t[2] - 64))
        self._grid_size_cache[key] = (best[0], best[1])
        return best[0], best[1]

    # -------------------- Template classifier --------------------

    def _load_assets(self) -> Dict[str, List["np.ndarray"]]:
        if np is None:
            return {}
        assets: Dict[str, List["np.ndarray"]] = {}
        roots = [os.path.join(os.getcwd(), "data", "assets")]

        file_path = globals().get("__file__")
        if isinstance(file_path, str) and file_path:
            roots.append(os.path.join(os.path.dirname(os.path.abspath(file_path)), "data", "assets"))

        try:
            import grid_adventure  # type: ignore

            pkg_root = os.path.dirname(getattr(grid_adventure, "__file__", "") or "")
            if pkg_root:
                roots.append(os.path.join(pkg_root, "assets"))
        except Exception:
            pass

        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                folder = os.path.basename(dirpath).lower()
                label = self._canonical_label(folder)
                if label is None:
                    continue
                for fname in filenames:
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    p = os.path.join(dirpath, fname)
                    img = self._read_image(p)
                    if img is not None:
                        assets.setdefault(label, []).append(img)

        # Resource API fallback for packaged environments.
        try:
            from importlib import resources as _resources

            root = _resources.files("grid_adventure").joinpath("assets")

            def _walk(node) -> None:
                try:
                    entries = list(node.iterdir())
                except Exception:
                    return
                for ent in entries:
                    try:
                        if ent.is_dir():
                            _walk(ent)
                            continue
                    except Exception:
                        continue
                    name = str(getattr(ent, "name", "")).lower()
                    if not name.endswith((".png", ".jpg", ".jpeg")):
                        continue
                    folder = str(getattr(getattr(ent, "parent", None), "name", "")).lower()
                    label = self._canonical_label(folder)
                    if label is None:
                        continue
                    try:
                        data = ent.read_bytes()
                    except Exception:
                        continue
                    img = self._read_image_bytes(data)
                    if img is not None:
                        assets.setdefault(label, []).append(img)

            _walk(root)
        except Exception:
            pass

        return assets

    def _canonical_label(self, raw: str) -> Optional[str]:
        n = str(raw).lower()
        if n in ("human", "agent", "player"):
            return "agent"
        if "gem" in n or "core" in n:
            return "gem"
        if "coin" in n:
            return "coin"
        if "key" in n:
            return "key"
        if "boots" in n or "speed" in n:
            return "boots"
        if "ghost" in n or "phasing" in n:
            return "ghost"
        if "shield" in n:
            return "shield"
        if "box" in n:
            return "box"
        if "exit" in n:
            return "exit"
        if "opened" in n or ("door" in n and "open" in n):
            return "door_open"
        if "locked" in n or ("door" in n and "lock" in n):
            return "door_locked"
        if "wall" in n:
            return "wall"
        if "lava" in n:
            return "lava"
        if "floor" in n or "ground" in n or "path" in n:
            return "floor"
        return None

    def _read_image(self, path: str):
        if np is None or not os.path.isfile(path):
            return None
        try:
            from PIL import Image

            with Image.open(path) as im:
                return np.asarray(im.convert("RGBA"))
        except Exception:
            return None

    def _read_image_bytes(self, data: bytes):
        if np is None:
            return None
        try:
            from PIL import Image
            import io

            with Image.open(io.BytesIO(data)) as im:
                return np.asarray(im.convert("RGBA"))
        except Exception:
            return None

    def _resize_image(self, img: "np.ndarray", w: int, h: int) -> "np.ndarray":
        if img.shape[0] == h and img.shape[1] == w:
            return img
        try:
            from PIL import Image

            im = Image.fromarray(img)
            im = im.resize((w, h), resample=Image.NEAREST)
            return np.asarray(im)
        except Exception:
            y_idx = (np.linspace(0, img.shape[0] - 1, h)).astype(int)
            x_idx = (np.linspace(0, img.shape[1] - 1, w)).astype(int)
            return img[y_idx][:, x_idx]

    def _build_templates(self, tile_h: int, tile_w: int) -> Dict[str, "np.ndarray"]:
        key = (tile_h, tile_w)
        if key in self._template_cache:
            return self._template_cache[key]

        all_labels = [
            "floor",
            "wall",
            "lava",
            "agent",
            "gem",
            "coin",
            "key",
            "boots",
            "ghost",
            "shield",
            "box",
            "door_locked",
            "door_open",
            "exit",
        ]
        feats: Dict[str, List["np.ndarray"]] = {k: [] for k in all_labels}

        resized: Dict[str, List["np.ndarray"]] = {}
        for label, imgs in self._assets.items():
            if label not in feats:
                continue
            for img in imgs:
                arr = self._resize_image(img, tile_w, tile_h)
                if arr.ndim != 3:
                    continue
                if arr.shape[2] == 3:
                    alpha = np.full((*arr.shape[:2], 1), 255, dtype=arr.dtype)
                    arr = np.concatenate([arr, alpha], axis=2)
                elif arr.shape[2] > 4:
                    arr = arr[..., :4]
                resized.setdefault(label, []).append(arr.astype(np.float32) / 255.0)

        floors = [x[..., :3] for x in resized.get("floor", [])]
        if not floors:
            floors = [np.zeros((tile_h, tile_w, 3), dtype=np.float32)]

        for label in ("floor", "wall", "lava"):
            for rgba in resized.get(label, []):
                feats[label].append(rgba[..., :3].reshape(-1))

        for label in (
            "agent",
            "gem",
            "coin",
            "key",
            "boots",
            "ghost",
            "shield",
            "box",
            "door_locked",
            "door_open",
            "exit",
        ):
            for rgba in resized.get(label, []):
                rgb = rgba[..., :3]
                a = rgba[..., 3:4]
                coverage = float(np.mean(a > 0.02))
                if coverage > 0.95:
                    feats[label].append(rgb.reshape(-1))
                else:
                    for floor_rgb in floors[:8]:
                        comp = rgb * a + floor_rgb * (1.0 - a)
                        feats[label].append(comp.reshape(-1))

        out: Dict[str, "np.ndarray"] = {}
        for label, vecs in feats.items():
            if vecs:
                out[label] = np.stack(vecs, axis=0)
        self._template_cache[key] = out
        return out

    def _classify_tile(self, tile_img: "np.ndarray", tile_h: int, tile_w: int) -> Optional[str]:
        if np is None:
            return None
        key = tile_img.tobytes()
        if key in self._tile_cache:
            return self._tile_cache[key]

        bank = self._build_templates(tile_h, tile_w)
        if not bank:
            self._tile_cache[key] = None
            return None

        rgba = tile_img
        if rgba.ndim != 3:
            self._tile_cache[key] = None
            return None
        if rgba.shape[2] == 3:
            alpha = np.full((*rgba.shape[:2], 1), 255, dtype=rgba.dtype)
            rgba = np.concatenate([rgba, alpha], axis=2)
        elif rgba.shape[2] > 4:
            rgba = rgba[..., :4]

        x = (rgba[..., :3].astype(np.float32) / 255.0).reshape(1, -1)
        best_label = None
        best_score = float("inf")
        second = float("inf")
        label_scores: Dict[str, float] = {}

        for label, mat in bank.items():
            d = np.mean((mat - x) ** 2, axis=1)
            score = float(np.min(d))
            label_scores[label] = score
            if score < best_score:
                second = best_score
                best_score = score
                best_label = label
            elif score < second:
                second = score

        thresholds = {
            "floor": 0.030,
            "wall": 0.040,
            "lava": 0.045,
            "agent": 0.110,
            "gem": 0.095,
            "coin": 0.095,
            "key": 0.105,
            "boots": 0.105,
            "ghost": 0.105,
            "shield": 0.105,
            "box": 0.100,
            "door_locked": 0.115,
            "door_open": 0.115,
            "exit": 0.120,
        }

        out = None
        if best_label is not None:
            thr = thresholds.get(best_label, 0.10)
            if best_score <= thr:
                out = best_label
            elif best_label in ("agent", "exit", "door_locked", "door_open"):
                if best_score <= thr * 1.45 and best_score * 1.18 < second:
                    out = best_label

        if out == "floor":
            floor_score = label_scores.get("floor")
            exit_score = label_scores.get("exit")
            if floor_score is not None and exit_score is not None:
                if floor_score >= 0.020 and exit_score <= floor_score * 1.16 and exit_score <= 0.050:
                    out = "exit"

        self._tile_cache[key] = out
        return out

    def _heuristic_tile(self, tile_img: "np.ndarray") -> Optional[str]:
        rgb = tile_img[..., :3]
        avg = rgb.mean(axis=(0, 1))
        r, g, b = float(avg[0]), float(avg[1]), float(avg[2])
        if r < 35 and g < 35 and b < 35:
            return "wall"
        if r > 120 and r > 1.3 * g and r > 1.3 * b:
            return "lava"
        if g > max(r, b) and g > 100:
            return "exit"
        return None

    # -------------------- State construction --------------------

    def _make_empty_gridstate(self, grid_w: int, grid_h: int) -> Optional[GridState]:
        try:
            from grid_adventure.entities import FloorEntity
            from grid_adventure.movements import MOVEMENTS
            from grid_adventure.objectives import OBJECTIVES

            movement = MOVEMENTS.get("cardinal") or MOVEMENTS.get("default")
            objective = OBJECTIVES.get("collect_gems_and_exit") or OBJECTIVES.get("default")
            state = GridState(width=grid_w, height=grid_h, movement=movement, objective=objective)
            for x in range(grid_w):
                for y in range(grid_h):
                    state.add((x, y), FloorEntity())
            return state
        except Exception:
            return None
