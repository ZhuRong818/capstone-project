from collections import deque
from typing import Dict, List, Optional, Set, Tuple

try:
    from grid_adventure.constants import COIN_REWARD, FLOOR_COST, HAZARD_DAMAGE
except Exception:
    COIN_REWARD = 10
    FLOOR_COST = 1
    HAZARD_DAMAGE = 1

from grid_adventure.grid import GridState
from grid_adventure.step import Action


Pos = Tuple[int, int]


class Agent:
    """
    Task 1 GridState-only agent.

    Design goals:
    - No hardcoded layouts
    - Generalizes to hidden cases
    - Handles: gems, exit, key/door, shield/lava, ghost/phasing, boots, boxes
    - Conservative optional-coin collection
    - Anti-loop fallback exploration
    """

    def __init__(self):
        self._mem_keys: int = 0
        self._ghost_turns_mem: int = 0
        self._collect_optional_coins: bool = False
        self._visit_counts: Dict[Pos, int] = {}
        self._recent_positions = deque(maxlen=8)

    # =========================================================
    # Public API
    # =========================================================
    def step(self, state: GridState) -> Action:
        if not isinstance(state, GridState):
            return Action.WAIT

        self._tick_memory()

        info = self._scan_grid(state)
        if info is None or info["agent"] is None:
            return Action.WAIT

        pos = info["agent"]
        self._remember_position(pos)

        # Pick up items when standing on them.
        if (
            pos in info["gems"]
            or pos in info["keys"]
            or pos in info["shields"]
            or pos in info["ghosts"]
            or pos in info["boots"]
            or (pos in info["coins"] and self._should_pick_coin_now(info))
        ):
            if pos in info["keys"]:
                self._mem_keys += 1
            if pos in info["ghosts"]:
                self._ghost_turns_mem = max(self._ghost_turns_mem, 6)
            return Action.PICK_UP

        # Unlock adjacent locked door when possible.
        if self._can_use_key_now(state, info):
            return Action.USE_KEY

        act = self._plan_action(state, info)
        return act if act is not None else Action.WAIT

    # =========================================================
    # Planner
    # =========================================================
    def _plan_action(self, state: GridState, info: Dict) -> Optional[Action]:
        start = info["agent"]
        if start is None:
            return None

        have_key = self._has_key(state) or self._mem_keys > 0
        have_shield = self._has_shield(state)
        have_ghost = self._has_ghost(state) or self._ghost_turns_mem > 0

        primary_targets = set(info["gems"]) if info["gems"] else set(info["exits"])

        # Reset optional-coin mode whenever required progress still exists.
        if info["gems"] or not info["coins"]:
            self._collect_optional_coins = False

        # 1) Optional coins, but only if worthwhile and no gems remain.
        if not info["gems"] and info["coins"]:
            should_collect = self._collect_optional_coins or self._should_collect_optional_coins(
                start,
                set(info["coins"]),
                set(info["exits"]),
                info,
                have_key,
                have_shield,
                have_ghost,
            )
            if should_collect:
                act = self._select_path_action(
                    state,
                    start,
                    set(info["coins"]),
                    info,
                    have_key,
                    have_shield,
                    have_ghost,
                )
                if act is not None:
                    self._collect_optional_coins = True
                    return act

        # 2) If shield is clearly needed before reaching the main goal, get shield first.
        if self._should_prioritize_shield(
            state,
            start,
            primary_targets,
            set(info["shields"]),
            info,
            have_key,
            have_shield,
            have_ghost,
        ):
            act = self._select_path_action(
                state,
                start,
                set(info["shields"]),
                info,
                have_key,
                False,
                have_ghost,
            )
            if act is not None:
                return act

        # 3) Try to solve the real objective directly.
        if primary_targets:
            direct = self._select_path_action(
                state,
                start,
                primary_targets,
                info,
                have_key,
                have_shield,
                have_ghost,
            )
            if direct is not None:
                return direct

        # 4) If we have a key, move next to a locked door frontier.
        if have_key and info["locked_doors"]:
            door_fronts = self._door_frontier_targets(info, have_shield, have_ghost)
            if door_fronts:
                act = self._select_path_action(
                    state,
                    start,
                    door_fronts,
                    info,
                    have_key,
                    have_shield,
                    have_ghost,
                )
                if act is not None:
                    return act

        # 5) If doors exist and we do not have key, get a key.
        if info["locked_doors"] and not have_key and info["keys"]:
            act = self._select_path_action(
                state,
                start,
                set(info["keys"]),
                info,
                False,
                have_shield,
                have_ghost,
            )
            if act is not None:
                return act

        # 6) If lava exists and no shield, try shield.
        if info["lava"] and not have_shield and info["shields"]:
            act = self._select_path_action(
                state,
                start,
                set(info["shields"]),
                info,
                have_key,
                False,
                have_ghost,
            )
            if act is not None:
                return act

        # 7) If normal routing seems blocked, try ghost.
        if not have_ghost and info["ghosts"]:
            act = self._select_path_action(
                state,
                start,
                set(info["ghosts"]),
                info,
                have_key,
                have_shield,
                False,
            )
            if act is not None:
                return act

        # 8) Boots are a nice-to-have fallback.
        if info["boots"]:
            act = self._select_path_action(
                state,
                start,
                set(info["boots"]),
                info,
                have_key,
                have_shield,
                have_ghost,
            )
            if act is not None:
                return act

        # 9) Anti-loop fallback exploration.
        return self._nearest_safe_move(start, info, have_key, have_shield, have_ghost)

    # =========================================================
    # Path selection
    # =========================================================
    def _select_path_action(
        self,
        state: GridState,
        start: Pos,
        targets: Set[Pos],
        info: Dict,
        have_key: bool,
        have_shield: bool,
        have_ghost: bool,
    ) -> Optional[Action]:
        safe_action = self._bfs_first_action(start, targets, info, have_key, have_shield, have_ghost)

        # If lava is already safe or absent, use normal path.
        if have_shield or not info["lava"]:
            return safe_action

        # Otherwise consider a controlled lava-crossing path.
        lava_budget = self._lava_cross_budget(state)
        if lava_budget <= 0:
            return safe_action

        safe_dist = self._bfs_distance(start, targets, info, have_key, have_shield, have_ghost)
        risky_action, risky_dist, risky_hazards = self._bfs_first_action_budgeted_lava(
            start,
            targets,
            info,
            have_key,
            have_ghost,
            lava_budget,
        )

        if risky_action is None:
            return safe_action

        # If no safe path exists, sometimes risk is worth it.
        if safe_action is None or safe_dist is None:
            if targets != set(info["exits"]) and risky_hazards is not None and risky_hazards * 2 > lava_budget:
                return None
            return risky_action

        # Heuristic: accept lava route if significantly shorter.
        if risky_dist is not None and risky_hazards is not None:
            if risky_dist + 2 * risky_hazards <= safe_dist:
                return risky_action

        return safe_action

    # =========================================================
    # BFS helpers
    # =========================================================
    def _bfs_first_action(
        self,
        start: Pos,
        targets: Set[Pos],
        info: Dict,
        have_key: bool,
        have_shield: bool,
        have_ghost: bool,
    ) -> Optional[Action]:
        if not targets:
            return None
        if start in targets:
            return Action.PICK_UP

        q = deque([start])
        seen = {start}
        parent: Dict[Pos, Tuple[Pos, Action]] = {}

        while q:
            cur = q.popleft()
            if cur in targets:
                return self._reconstruct_first_action(start, cur, parent)

            for nxt, act in self._neighbors(cur, info, have_key, have_shield, have_ghost):
                if nxt in seen:
                    continue
                seen.add(nxt)
                parent[nxt] = (cur, act)
                q.append(nxt)

        return None

    def _bfs_distance(
        self,
        start: Pos,
        targets: Set[Pos],
        info: Dict,
        have_key: bool,
        have_shield: bool,
        have_ghost: bool,
    ) -> Optional[int]:
        if not targets:
            return None
        if start in targets:
            return 0

        q = deque([(start, 0)])
        seen = {start}

        while q:
            cur, dist = q.popleft()
            for nxt, _ in self._neighbors(cur, info, have_key, have_shield, have_ghost):
                if nxt in seen:
                    continue
                if nxt in targets:
                    return dist + 1
                seen.add(nxt)
                q.append((nxt, dist + 1))
        return None

    def _reconstruct_first_action(
        self,
        start: Pos,
        goal: Pos,
        parent: Dict[Pos, Tuple[Pos, Action]],
    ) -> Optional[Action]:
        cur = goal
        while cur in parent and parent[cur][0] != start:
            cur = parent[cur][0]
        if cur not in parent:
            return None
        return parent[cur][1]

    # =========================================================
    # Neighbors / movement model
    # =========================================================
    def _neighbors(
        self,
        pos: Pos,
        info: Dict,
        have_key: bool,
        have_shield: bool,
        have_ghost: bool,
    ):
        x, y = pos
        dirs = [
            ((x + 1, y), Action.RIGHT, (1, 0)),
            ((x - 1, y), Action.LEFT, (-1, 0)),
            ((x, y + 1), Action.DOWN, (0, 1)),
            ((x, y - 1), Action.UP, (0, -1)),
        ]

        for nxt, act, (dx, dy) in dirs:
            nx, ny = nxt
            if not self._inside(nx, ny, info):
                continue

            # Ghost can phase through most blockers; still respect lava if no shield.
            if have_ghost:
                if not have_shield and nxt in info["lava"]:
                    continue
                yield nxt, act
                continue

            if nxt in info["walls"]:
                continue

            # Locked doors are not traversable until explicitly unlocked.
            if nxt in info["locked_doors"]:
                continue

            if nxt in info["lava"] and not have_shield:
                continue

            # Local pushable-box handling.
            if nxt in info["boxes"]:
                bx, by = nx + dx, ny + dy
                if not self._inside(bx, by, info):
                    continue
                if (bx, by) in info["walls"]:
                    continue
                if (bx, by) in info["locked_doors"]:
                    continue
                if (bx, by) in info["boxes"]:
                    continue
                if (bx, by) in info["lava"] and not have_shield:
                    continue
                yield nxt, act
                continue

            yield nxt, act

    # =========================================================
    # Lava-budget BFS
    # =========================================================
    def _bfs_first_action_budgeted_lava(
        self,
        start: Pos,
        targets: Set[Pos],
        info: Dict,
        have_key: bool,
        have_ghost: bool,
        budget: int,
    ) -> Tuple[Optional[Action], Optional[int], Optional[int]]:
        if not targets:
            return None, None, None
        if start in targets:
            return Action.PICK_UP, 0, 0

        start_state = (start, 0)
        q = deque([start_state])
        seen = {start_state}
        parent: Dict[Tuple[Pos, int], Tuple[Tuple[Pos, int], Action]] = {}
        dist: Dict[Tuple[Pos, int], int] = {start_state: 0}

        while q:
            cur, used = q.popleft()
            cur_state = (cur, used)

            if cur in targets:
                first = self._reconstruct_first_action_budgeted(start_state, cur_state, parent)
                return first, dist[cur_state], used

            for nxt, act, is_hazard in self._neighbors_budgeted_lava(cur, info, have_key, have_ghost):
                nxt_used = used + (1 if is_hazard else 0)
                if nxt_used > budget:
                    continue
                nxt_state = (nxt, nxt_used)
                if nxt_state in seen:
                    continue
                seen.add(nxt_state)
                parent[nxt_state] = (cur_state, act)
                dist[nxt_state] = dist[cur_state] + 1
                q.append(nxt_state)

        return None, None, None

    def _reconstruct_first_action_budgeted(
        self,
        start_state: Tuple[Pos, int],
        goal_state: Tuple[Pos, int],
        parent: Dict[Tuple[Pos, int], Tuple[Tuple[Pos, int], Action]],
    ) -> Optional[Action]:
        cur = goal_state
        while cur in parent and parent[cur][0] != start_state:
            cur = parent[cur][0]
        if cur not in parent:
            return None
        return parent[cur][1]

    def _neighbors_budgeted_lava(
        self,
        pos: Pos,
        info: Dict,
        have_key: bool,
        have_ghost: bool,
    ):
        x, y = pos
        dirs = [
            ((x + 1, y), Action.RIGHT, (1, 0)),
            ((x - 1, y), Action.LEFT, (-1, 0)),
            ((x, y + 1), Action.DOWN, (0, 1)),
            ((x, y - 1), Action.UP, (0, -1)),
        ]

        for nxt, act, (dx, dy) in dirs:
            nx, ny = nxt
            if not self._inside(nx, ny, info):
                continue

            if have_ghost:
                yield nxt, act, nxt in info["lava"]
                continue

            if nxt in info["walls"]:
                continue
            if nxt in info["locked_doors"]:
                continue

            if nxt in info["boxes"]:
                bx, by = nx + dx, ny + dy
                if not self._inside(bx, by, info):
                    continue
                if (bx, by) in info["walls"]:
                    continue
                if (bx, by) in info["locked_doors"]:
                    continue
                if (bx, by) in info["boxes"]:
                    continue
                yield nxt, act, nxt in info["lava"]
                continue

            yield nxt, act, nxt in info["lava"]

    # =========================================================
    # Strategic helpers
    # =========================================================
    def _should_prioritize_shield(
        self,
        state: GridState,
        start: Pos,
        targets: Set[Pos],
        shields: Set[Pos],
        info: Dict,
        have_key: bool,
        have_shield: bool,
        have_ghost: bool,
    ) -> bool:
        if have_shield or not shields or not targets or not info["lava"]:
            return False

        safe_dist = self._bfs_distance(start, targets, info, have_key, have_shield, have_ghost)
        if safe_dist is not None:
            return False

        lava_budget = self._lava_cross_budget(state)
        if lava_budget <= 0:
            return True

        risky_action, _, risky_hazards = self._bfs_first_action_budgeted_lava(
            start,
            targets,
            info,
            have_key,
            have_ghost,
            lava_budget,
        )

        if risky_action is None or risky_hazards is None:
            return True
        if risky_hazards * 2 > lava_budget:
            return True
        return False

    def _door_frontier_targets(self, info: Dict, have_shield: bool, have_ghost: bool) -> Set[Pos]:
        out: Set[Pos] = set()
        for x, y in info["locked_doors"]:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if not self._inside(nx, ny, info):
                    continue
                pos = (nx, ny)
                if pos in info["walls"]:
                    continue
                if pos in info["boxes"] and not have_ghost:
                    continue
                if pos in info["lava"] and not have_shield:
                    continue
                out.add(pos)
        return out

    def _nearest_safe_move(
        self,
        start: Pos,
        info: Dict,
        have_key: bool,
        have_shield: bool,
        have_ghost: bool,
    ) -> Optional[Action]:
        q = deque([start])
        seen = {start}
        parent: Dict[Pos, Tuple[Pos, Action]] = {}
        dist: Dict[Pos, int] = {start: 0}
        first_hop: Dict[Pos, Pos] = {}

        best_target: Optional[Pos] = None
        best_score: Optional[Tuple[int, int, int, int]] = None

        while q:
            cur = q.popleft()
            cur_dist = dist[cur]

            if cur != start:
                recent_penalty = sum(1 for p in self._recent_positions if p == cur)
                visits = self._visit_counts.get(cur, 0)
                reverse_penalty = 0
                hop = first_hop.get(cur)
                if len(self._recent_positions) >= 2 and hop == self._recent_positions[-2]:
                    reverse_penalty = 1

                # Lower score is better.
                score = (
                    visits + 3 * recent_penalty + 6 * reverse_penalty,
                    reverse_penalty,
                    recent_penalty,
                    -cur_dist,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_target = cur

            for nxt, act in self._neighbors(cur, info, have_key, have_shield, have_ghost):
                if nxt in seen:
                    continue
                seen.add(nxt)
                parent[nxt] = (cur, act)
                dist[nxt] = cur_dist + 1
                first_hop[nxt] = nxt if cur == start else first_hop[cur]
                q.append(nxt)

        if best_target is None:
            return None
        return self._reconstruct_first_action(start, best_target, parent)

    # =========================================================
    # Optional coin policy
    # =========================================================
    def _should_pick_coin_now(self, info: Dict) -> bool:
        # Pick coin when we're already in optional-coin mode or there are no gems left.
        return self._collect_optional_coins or not info["gems"]

    def _is_coin_corridor(self, coins: Set[Pos], exits: Set[Pos]) -> bool:
        if not coins or not exits:
            return False

        ys = {y for _, y in coins}
        if len(ys) == 1:
            y = next(iter(ys))
            xs = sorted(x for x, _ in coins)
            if xs == list(range(xs[0], xs[-1] + 1)):
                for ex, ey in exits:
                    if ey == y and (ex == xs[0] - 1 or ex == xs[-1] + 1):
                        return True

        xs = {x for x, _ in coins}
        if len(xs) == 1:
            x = next(iter(xs))
            ys2 = sorted(y for _, y in coins)
            if ys2 == list(range(ys2[0], ys2[-1] + 1)):
                for ex, ey in exits:
                    if ex == x and (ey == ys2[0] - 1 or ey == ys2[-1] + 1):
                        return True

        return False

    def _should_collect_optional_coins(
        self,
        start: Pos,
        coins: Set[Pos],
        exits: Set[Pos],
        info: Dict,
        have_key: bool,
        have_shield: bool,
        have_ghost: bool,
    ) -> bool:
        if not coins or not exits:
            return False

        exit_dist = self._bfs_distance(start, exits, info, have_key, have_shield, have_ghost)
        if exit_dist is None:
            return False

        coin_dist = self._bfs_distance(start, coins, info, have_key, have_shield, have_ghost)
        if coin_dist is None:
            return False

        # Corridor coins are often "on the way".
        if self._is_coin_corridor(coins, exits):
            return True

        # Single cheap coin detour is often worth it.
        if len(coins) == 1:
            return coin_dist <= exit_dist + 1

        finish_steps = self._best_optional_coin_finish_steps(
            start, coins, exits, info, have_key, have_shield, have_ghost
        )
        if finish_steps is None:
            return False

        extra_steps = finish_steps - exit_dist
        return len(coins) * COIN_REWARD > extra_steps * FLOOR_COST

    def _best_optional_coin_finish_steps(
        self,
        start: Pos,
        coins: Set[Pos],
        exits: Set[Pos],
        info: Dict,
        have_key: bool,
        have_shield: bool,
        have_ghost: bool,
    ) -> Optional[int]:
        coin_list = sorted(coins)
        if not coin_list or len(coin_list) > 6:
            return None

        start_to_coin = [
            self._bfs_distance(start, {coin}, info, have_key, have_shield, have_ghost)
            for coin in coin_list
        ]
        coin_to_exit = [
            self._bfs_distance(coin, exits, info, have_key, have_shield, have_ghost)
            for coin in coin_list
        ]

        pair_dist: List[List[Optional[int]]] = []
        for src in coin_list:
            row: List[Optional[int]] = []
            for dst in coin_list:
                if src == dst:
                    row.append(0)
                else:
                    row.append(self._bfs_distance(src, {dst}, info, have_key, have_shield, have_ghost))
            pair_dist.append(row)

        dp: Dict[Tuple[int, int], int] = {}
        for idx, dist in enumerate(start_to_coin):
            if dist is not None:
                dp[(1 << idx, idx)] = dist

        full_mask = (1 << len(coin_list)) - 1
        for mask in range(1, full_mask + 1):
            for last in range(len(coin_list)):
                cur = dp.get((mask, last))
                if cur is None:
                    continue
                for nxt in range(len(coin_list)):
                    if mask & (1 << nxt):
                        continue
                    step = pair_dist[last][nxt]
                    if step is None:
                        continue
                    key = (mask | (1 << nxt), nxt)
                    cand = cur + step
                    prev = dp.get(key)
                    if prev is None or cand < prev:
                        dp[key] = cand

        best: Optional[int] = None
        for last in range(len(coin_list)):
            cur = dp.get((full_mask, last))
            tail = coin_to_exit[last]
            if cur is None or tail is None:
                continue
            total = cur + tail
            if best is None or total < best:
                best = total
        return best

    # =========================================================
    # Inventory / state helpers
    # =========================================================
    def _agent_items(self, grid_state: GridState) -> List[object]:
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return []
        x, y = pos
        for obj in grid_state.grid[x][y]:
            n = self._appearance_name(obj)
            is_agent = getattr(obj, "agent", None) is not None or n in ("agent", "human")
            if not is_agent:
                continue
            inv = list(getattr(obj, "inventory_list", None) or [])
            status = list(getattr(obj, "status_list", None) or [])
            return inv + status
        return []

    def _has_key(self, grid_state: GridState) -> bool:
        for item in self._agent_items(grid_state):
            if getattr(item, "key", None) is not None or self._appearance_name(item) == "key":
                return True
        return False

    def _has_shield(self, grid_state: GridState) -> bool:
        for item in self._agent_items(grid_state):
            if getattr(item, "immunity", None) is not None or self._appearance_name(item) == "shield":
                return True
        return False

    def _has_ghost(self, grid_state: GridState) -> bool:
        for item in self._agent_items(grid_state):
            if getattr(item, "phasing", None) is not None or self._appearance_name(item) == "ghost":
                return True
        return False

    def _can_use_key_now(self, grid_state: GridState, info: Dict) -> bool:
        pos = info["agent"]
        if pos is None:
            return False
        if not (self._has_key(grid_state) or self._mem_keys > 0):
            return False
        x, y = pos
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (x + dx, y + dy) in info["locked_doors"]:
                return True
        return False

    def _agent_health(self, grid_state: GridState) -> Optional[int]:
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return None
        x, y = pos
        for obj in grid_state.grid[x][y]:
            n = self._appearance_name(obj)
            is_agent = getattr(obj, "agent", None) is not None or n in ("agent", "human")
            if not is_agent:
                continue
            health = getattr(obj, "health", None)
            if health is not None:
                current = getattr(health, "current_health", None)
                if isinstance(current, (int, float)):
                    return int(current)
            for attr in ("current_health", "hp"):
                value = getattr(obj, attr, None)
                if isinstance(value, (int, float)):
                    return int(value)
        return None

    def _lava_cross_budget(self, grid_state: GridState) -> int:
        if self._has_shield(grid_state):
            return 10**9
        health = self._agent_health(grid_state)
        if health is None:
            return 0
        damage = max(1, int(HAZARD_DAMAGE))
        return max(0, (int(health) - 1) // damage)

    # =========================================================
    # Grid scanning
    # =========================================================
    def _scan_grid(self, grid_state: GridState) -> Optional[Dict]:
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
        except Exception:
            return None

        agent_pos = None
        gems: Set[Pos] = set()
        exits: Set[Pos] = set()
        coins: Set[Pos] = set()
        keys: Set[Pos] = set()
        shields: Set[Pos] = set()
        ghosts: Set[Pos] = set()
        boots: Set[Pos] = set()
        locked_doors: Set[Pos] = set()
        walls: Set[Pos] = set()
        lava: Set[Pos] = set()
        boxes: Set[Pos] = set()

        for x in range(width):
            for y in range(height):
                for obj in grid_state.grid[x][y]:
                    n = self._appearance_name(obj)

                    if getattr(obj, "agent", None) is not None or n in ("agent", "human"):
                        agent_pos = (x, y)

                    elif getattr(obj, "requirable", None) is not None or n in ("gem", "core"):
                        gems.add((x, y))

                    elif getattr(obj, "exit", None) is not None or n == "exit":
                        exits.add((x, y))

                    elif getattr(obj, "rewardable", None) is not None and n == "coin":
                        coins.add((x, y))

                    elif getattr(obj, "key", None) is not None or n == "key":
                        keys.add((x, y))

                    elif getattr(obj, "immunity", None) is not None or n == "shield":
                        shields.add((x, y))

                    elif getattr(obj, "phasing", None) is not None or n == "ghost":
                        ghosts.add((x, y))

                    elif getattr(obj, "speed", None) is not None or n == "boots":
                        boots.add((x, y))

                    elif getattr(obj, "locked", None) is not None or n in ("locked", "door_locked"):
                        locked_doors.add((x, y))

                    elif getattr(obj, "pushable", None) is not None or n == "box":
                        boxes.add((x, y))

                    else:
                        is_wall = (
                            n == "wall"
                            or (
                                getattr(obj, "blocking", None) is not None
                                and getattr(obj, "locked", None) is None
                                and getattr(obj, "pushable", None) is None
                                and n not in ("door", "opened", "door_open")
                            )
                        )
                        if is_wall:
                            walls.add((x, y))

                        if getattr(obj, "damage", None) is not None or n in ("lava", "spike"):
                            lava.add((x, y))

        return {
            "width": width,
            "height": height,
            "agent": agent_pos,
            "gems": gems,
            "exits": exits,
            "coins": coins,
            "keys": keys,
            "shields": shields,
            "ghosts": ghosts,
            "boots": boots,
            "locked_doors": locked_doors,
            "walls": walls,
            "lava": lava,
            "boxes": boxes,
        }

    # =========================================================
    # Basic helpers
    # =========================================================
    def _find_agent_pos(self, grid_state: GridState) -> Optional[Pos]:
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
        except Exception:
            return None

        for x in range(width):
            for y in range(height):
                for obj in grid_state.grid[x][y]:
                    n = self._appearance_name(obj)
                    if getattr(obj, "agent", None) is not None or n in ("agent", "human"):
                        return (x, y)
        return None

    def _remember_position(self, pos: Pos) -> None:
        self._visit_counts[pos] = self._visit_counts.get(pos, 0) + 1
        self._recent_positions.append(pos)

    def _appearance_name(self, obj) -> str:
        return str(getattr(getattr(obj, "appearance", None), "name", "")).lower()

    def _inside(self, x: int, y: int, info: Dict) -> bool:
        return 0 <= x < info["width"] and 0 <= y < info["height"]

    def _tick_memory(self) -> None:
        if self._ghost_turns_mem > 0:
            self._ghost_turns_mem -= 1
        if self._ghost_turns_mem < 0:
            self._ghost_turns_mem = 0