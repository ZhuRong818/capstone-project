from collections import deque
from typing import Dict, List, Optional, Tuple

from grid_adventure.env import ImageObservation
from grid_adventure.grid import GridState
from grid_adventure.step import Action


class Agent:
    """Task 1 agent optimized for low-latency per-step planning."""

    def __init__(self):
        self._dir_cycle = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        self._dir_idx = 0

        # Movement/stability memory.
        self._last_step_agent_pos: Optional[Tuple[int, int]] = None
        self._last_step_action: Optional[Action] = None
        self._stuck_steps: int = 0
        self._visit_counts: Dict[Tuple[int, int], int] = {}

        # Interaction memory.
        self._expected_pickup_pos: Optional[Tuple[int, int]] = None
        self._expected_pickup_kind: Optional[str] = None
        self._mem_keys: int = 0
        self._mem_shield: bool = False
        self._mem_ghost: bool = False
        self._ghost_turn_duration: int = 5
        self._mem_ghost_turns: int = 0

        # Episode/turn tracking.
        self._last_seen_turn: Optional[int] = None

    # -------------------- Public API --------------------

    def step(self, state: GridState | ImageObservation) -> Action:
        # Task 1 should be GridState; keep a safe guard for accidental image inputs.
        if self._is_image_observation(state):
            return Action.WAIT

        grid_state = self._extract_gridstate(state)
        if grid_state is None:
            return self._cycle_fallback()

        self._tick_turn_memory(grid_state)

        cur_pos = self._find_agent_pos(grid_state)
        if cur_pos is not None:
            self._visit_counts[cur_pos] = self._visit_counts.get(cur_pos, 0) + 1

        legal = self._legal_actions(grid_state)
        legal_set = set(legal)

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
        has_boots = False
        has_ghost = False
        has_shield = False

        gems = set()
        exits = set()
        coins = set()
        keys_on_ground = set()
        boots_on_ground = set()
        ghosts_on_ground = set()
        shields_on_ground = set()
        walls = set()
        boxes = set()
        locked_doors = set()
        lava = set()

        for x in range(width):
            for y in range(height):
                for obj in grid_state.grid[x][y]:
                    name = self._appearance_name(obj)
                    is_agent = getattr(obj, "agent", None) is not None or name in ("agent", "human")
                    if is_agent:
                        if agent_pos is None:
                            agent_pos = (x, y)
                        inv = list(getattr(obj, "inventory_list", None) or [])
                        status = list(getattr(obj, "status_list", None) or [])
                        for item in inv + status:
                            iname = self._appearance_name(item)
                            if getattr(item, "key", None) is not None or iname == "key":
                                keys_count += 1
                            if getattr(item, "speed", None) is not None or iname == "boots":
                                has_boots = True
                            if getattr(item, "phasing", None) is not None or iname == "ghost":
                                has_ghost = True
                            if getattr(item, "immunity", None) is not None or iname == "shield":
                                has_shield = True

                    if getattr(obj, "requirable", None) is not None or name in ("gem", "core"):
                        gems.add((x, y))
                    if getattr(obj, "exit", None) is not None or name == "exit":
                        exits.add((x, y))
                    if getattr(obj, "rewardable", None) is not None and name == "coin":
                        coins.add((x, y))
                    if getattr(obj, "key", None) is not None or name == "key":
                        keys_on_ground.add((x, y))
                    if getattr(obj, "speed", None) is not None or name == "boots":
                        boots_on_ground.add((x, y))
                    if getattr(obj, "phasing", None) is not None or name == "ghost":
                        ghosts_on_ground.add((x, y))
                    if getattr(obj, "immunity", None) is not None or name == "shield":
                        shields_on_ground.add((x, y))
                    if getattr(obj, "locked", None) is not None or name in ("locked", "door_locked"):
                        locked_doors.add((x, y))
                    if getattr(obj, "pushable", None) is not None or name == "box":
                        boxes.add((x, y))
                    is_wall = name == "wall" or (
                        getattr(obj, "blocking", None) is not None
                        and getattr(obj, "locked", None) is None
                        and getattr(obj, "pushable", None) is None
                        and name not in ("door", "opened", "door_open")
                    )
                    if is_wall:
                        walls.add((x, y))
                    if getattr(obj, "damage", None) is not None or name == "lava":
                        lava.add((x, y))

        if agent_pos is None:
            return None

        # Memory fallback for parse noise on inventory/status.
        keys_count += max(0, int(self._mem_keys))
        has_shield = has_shield or self._mem_shield
        has_ghost = has_ghost or self._mem_ghost or self._mem_ghost_turns > 0
        _ = has_boots

        # Current-tile interaction first.
        if (
            agent_pos in gems
            or agent_pos in coins
            or agent_pos in keys_on_ground
            or agent_pos in boots_on_ground
            or agent_pos in ghosts_on_ground
            or agent_pos in shields_on_ground
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

        # Door interaction.
        if keys_count > 0:
            ax, ay = agent_pos
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (ax + dx, ay + dy) in locked_doors:
                    if self._mem_keys > 0:
                        self._mem_keys -= 1
                    return Action.USE_KEY

        # Goal-conditioning helpers.
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
        need_shield_for_lava = bool(lava) and not has_shield and bool(shields_on_ground)
        powerups = set(shields_on_ground) | set(ghosts_on_ground) | set(boots_on_ground)

        blocked = set()
        if not has_ghost:
            blocked |= walls
            blocked |= locked_doors
        if not has_shield:
            blocked |= lava
        blocked.discard(agent_pos)

        # Ordered tries: required mechanics, required objective, then score, then exit.
        ordered_target_groups = []
        if need_key_for_door:
            ordered_target_groups.append(keys_on_ground)
        if door_adj_targets:
            ordered_target_groups.append(door_adj_targets)
        if need_shield_for_lava:
            ordered_target_groups.append(shields_on_ground)

        if gems:
            ordered_target_groups.append(gems)
            # If required objective appears blocked, try unlockers before giving up.
            if keys_on_ground:
                ordered_target_groups.append(keys_on_ground)
            if powerups:
                ordered_target_groups.append(powerups)
        else:
            # Optional reward before exit for better performance on coin maps.
            if coins:
                ordered_target_groups.append(coins)
            if exits:
                ordered_target_groups.append(exits)
            if keys_on_ground:
                ordered_target_groups.append(keys_on_ground)
            if powerups:
                ordered_target_groups.append(powerups)

        seen_groups = set()
        for targets in ordered_target_groups:
            if not targets:
                continue
            frozen = frozenset(targets)
            if frozen in seen_groups:
                continue
            seen_groups.add(frozen)
            action, found_target = self._bfs_first_action(agent_pos, list(targets), width, height, blocked)
            if action is not None:
                self._set_expected_pickup(
                    found_target,
                    keys_on_ground,
                    shields_on_ground,
                    boots_on_ground,
                    ghosts_on_ground,
                )
                return action

        self._expected_pickup_pos = None
        self._expected_pickup_kind = None
        return None

    # -------------------- Action filters / fallbacks --------------------

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

        return self._cycle_fallback()

    def _cycle_fallback(self) -> Action:
        act = self._dir_cycle[self._dir_idx]
        self._dir_idx = (self._dir_idx + 1) % len(self._dir_cycle)
        return act

    def _legal_actions(self, grid_state: GridState) -> List[Action]:
        legal: List[Action] = []
        if self._should_force_pickup(grid_state):
            legal.append(Action.PICK_UP)
        if self._can_use_key_now(grid_state):
            legal.append(Action.USE_KEY)
        for act in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            if self._is_move_candidate(grid_state, act):
                legal.append(act)
        if not legal:
            legal.append(Action.WAIT)
        return legal

    def _is_move_candidate(self, grid_state: GridState, act: Action) -> bool:
        if act not in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            return False
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return True
        npos = self._neighbor_pos(pos, act)
        if npos is None:
            return False
        nx, ny = npos
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

    # -------------------- Planner helpers --------------------

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

    # -------------------- Safety / anti-stuck --------------------

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

    def _is_lava_move(self, grid_state: GridState, act: Action) -> bool:
        if act not in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            return False
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return False
        npos = self._neighbor_pos(pos, act)
        if npos is None:
            return False
        nx, ny = npos
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

    # -------------------- State/introspection helpers --------------------

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

    # -------------------- Episode memory --------------------

    def _tick_turn_memory(self, state: GridState) -> None:
        turn = None
        try:
            maybe_turn = getattr(state, "turn", None)
            turn = int(maybe_turn) if maybe_turn is not None else None
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

    def _activate_ghost_memory(self) -> None:
        self._mem_ghost = True
        self._mem_ghost_turns = max(self._mem_ghost_turns, self._ghost_turn_duration + 1)

    # -------------------- Input handling --------------------

    def _is_image_observation(self, state) -> bool:
        if isinstance(state, dict) and "image" in state:
            return True
        return hasattr(state, "image")

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
