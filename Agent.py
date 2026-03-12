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
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

if nn is not None:
    class _TinyTileNet(nn.Module):
        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 96, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(96, n_classes)

        def forward(self, x):
            z = self.features(x).flatten(1)
            return self.head(z)
else:
    class _TinyTileNet:
        def __init__(self, n_classes: int):
            self.n_classes = n_classes


class Agent:
    """Grid Adventure: Variant 1 agent template.

    This implementation reasons on structured GridState whenever available.
    For ImageObservation, it parses the image into an approximate GridState
    using local assets from data/assets, then plans an action.
    """

    def __init__(self):
        """Initialize your agent."""
        self._rng = random.Random(0)
        self._dir_cycle = [Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP]
        self._dir_idx = 0
        self._planner = None

        # Perception caches
        self._assets = self._load_assets()
        self._template_cache: Dict[Tuple[int, int], Dict[str, "np.ndarray"]] = {}
        self._tile_cache: Dict[bytes, Optional[str]] = {}
        self._grid_size_cache: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self._ml_model = None
        self._ml_labels: List[str] = []
        self._ml_image_size: int = 32
        self._ml_confidence_threshold: float = 0.60
        self._ml_class_thresholds: Dict[str, float] = {
            "key": 0.82,
            "door_locked": 0.86,
            "door_open": 0.82,
            "shield": 0.84,
            "ghost": 0.84,
            "boots": 0.84,
            "agent": 0.72,
        }
        self._use_ml = os.environ.get("AGENT_USE_ML", "0") == "1"
        if self._use_ml:
            self._load_ml_model()
        self._temporal_unstable_labels = {
            "key",
            "door_locked",
            "door_open",
            "shield",
            "ghost",
            "boots",
        }
        self._temporal_min_streak = max(1, int(os.environ.get("AGENT_TEMPORAL_MIN_STREAK", "1")))
        self._tile_label_streaks: Dict[Tuple[int, int], Tuple[str, int]] = {}
        self._last_label_counts: Dict[str, int] = {}
        self._last_label_grid: List[List[str]] = []

        # Stability memory
        self._last_agent_pos: Optional[Tuple[int, int]] = None
        self._last_good_state: Optional[GridState] = None
        self._last_good_turn: Optional[int] = None
        self._last_flip_y: Optional[bool] = None
        self._consecutive_reuse: int = 0
        self._last_step_agent_pos: Optional[Tuple[int, int]] = None
        self._last_step_action: Optional[Action] = None
        self._stuck_steps: int = 0
        self._expected_pickup_pos: Optional[Tuple[int, int]] = None
        self._expected_pickup_kind: Optional[str] = None
        self._mem_boots: bool = False
        self._mem_ghost: bool = False
        self._ghost_turn_duration: int = max(1, int(os.environ.get("AGENT_GHOST_TURNS", "5")))
        # We use turn-bounded ghost memory to avoid stale phasing assumptions.
        self._mem_ghost_turns: int = 0
        self._mem_shield: bool = False
        self._mem_keys: int = 0
        self._last_seen_turn: Optional[int] = None
        self._visit_counts: Dict[Tuple[int, int], int] = {}
        self._exit_memory_ttl: int = max(1, int(os.environ.get("AGENT_EXIT_MEMORY_TTL", "80")))
        self._exit_memory: Dict[Tuple[int, int], int] = {}
        # Target persistence is optional; keep it off by default to avoid stale-goal loops.
        self._target_memory_ttl: int = max(0, int(os.environ.get("AGENT_TARGET_TTL", "0")))
        self._target_memory: Dict[str, Dict[Tuple[int, int], int]] = {
            "gem": {},
            "exit": {},
            "key": {},
            "boots": {},
            "ghost": {},
            "shield": {},
            "door_locked": {},
        }

        # Optional debug dump mode (AGENT_DEBUG=1).
        self._debug = os.environ.get("AGENT_DEBUG", "0") == "1"
        self._debug_seq = 0
        self._debug_run_id = f"{os.getpid()}_{int(time.time() * 1000)}"
        self._debug_dir = os.path.join(os.getcwd(), "debug_frames")
        if self._debug:
            try:
                os.makedirs(self._debug_dir, exist_ok=True)
            except Exception:
                self._debug = False

        # Reuse your Task 1 planner when possible.
        try:
            from Agent import Agent as GridAgent  # type: ignore

            # Avoid recursive self-instantiation when Agent.py re-exports AgentImage.Agent.
            self._planner = None if GridAgent is Agent else GridAgent()
        except Exception:
            self._planner = None
        # In image mode we can use a lightweight legality filter to avoid
        # repeated deep-copy simulation costs that cause wall-time timeouts.
        self._fast_image_legality = os.environ.get("AGENT_FAST_IMAGE_LEGAL", "1") == "1"
        self._step_input_is_image: bool = False

    def _load_ml_model(self) -> None:
        if torch is None or nn is None:
            return
        model_path = os.path.join(os.getcwd(), "data", "tile_model.pt")
        if not os.path.isfile(model_path):
            return
        try:
            ckpt = torch.load(model_path, map_location="cpu")
            labels = list(ckpt.get("labels") or [])
            if not labels:
                return
            image_size = int(ckpt.get("image_size", 32))
            state_dict = ckpt.get("state_dict")
            if not isinstance(state_dict, dict):
                return
            model = _TinyTileNet(len(labels))
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            self._ml_model = model
            self._ml_labels = [str(x) for x in labels]
            self._ml_image_size = max(8, image_size)
        except Exception:
            self._ml_model = None
            self._ml_labels = []

    # -------------------- Public API --------------------

    def step(self, state: GridState | ImageObservation) -> Action:
        """Return the next action given the current environment state."""
        self._tick_turn_memory(state)
        input_is_structured_grid = isinstance(state, GridState)
        self._step_input_is_image = not input_is_structured_grid
        grid_state = self._extract_gridstate(state)
        if grid_state is None and self._is_image_observation(state):
            grid_state = self._parse_image_observation(state)

        if grid_state is not None:
            cur_pos = self._find_agent_pos(grid_state)
            if cur_pos is not None:
                self._visit_counts[cur_pos] = self._visit_counts.get(cur_pos, 0) + 1
            legal = self._legal_actions(grid_state)
            legal_set = set(legal)
            if self._should_force_pickup(grid_state) and Action.PICK_UP in legal_set:
                self._apply_expected_pickup_memory()
                self._expected_pickup_pos = None
                self._expected_pickup_kind = None
                self._dump_frame(state, Action.PICK_UP, legal, "forced_pickup", grid_state)
                return Action.PICK_UP

            act = self._reason_action(grid_state)
            if isinstance(act, Action) and act in legal_set and act != Action.WAIT:
                act = self._anti_stuck_adjust(act, grid_state, legal)
                act = self._avoid_lava_without_shield(act, grid_state, legal)
                self._dump_frame(state, act, legal, "reason", grid_state)
                return act

            # Raw-observation planner fallback can sometimes recover when parse drifts.
            if self._planner is not None:
                try:
                    raw_act = self._planner.step(state)
                    if isinstance(raw_act, Action) and raw_act in legal_set and raw_act != Action.WAIT:
                        raw_act = self._anti_stuck_adjust(raw_act, grid_state, legal)
                        raw_act = self._avoid_lava_without_shield(raw_act, grid_state, legal)
                        self._dump_frame(state, raw_act, legal, "planner_raw_legal", grid_state)
                        return raw_act
                except Exception:
                    pass

            if self._planner is not None and input_is_structured_grid:
                try:
                    act = self._planner.step(grid_state)
                    if isinstance(act, Action) and act in legal_set and act != Action.WAIT:
                        act = self._anti_stuck_adjust(act, grid_state, legal)
                        act = self._avoid_lava_without_shield(act, grid_state, legal)
                        self._dump_frame(state, act, legal, "planner_grid", grid_state)
                        return act
                except Exception:
                    pass

            fallback = self._safe_fallback(grid_state, legal)
            fallback = self._anti_stuck_adjust(fallback, grid_state, legal)
            fallback = self._avoid_lava_without_shield(fallback, grid_state, legal)
            self._dump_frame(state, fallback, legal, "fallback_grid", grid_state)
            return fallback

        if self._planner is not None:
            try:
                act = self._planner.step(state)
                if isinstance(act, Action) and act != Action.WAIT:
                    self._dump_frame(state, act, [], "planner_raw", None)
                    return act
            except Exception:
                pass

        fallback = self._safe_fallback(grid_state)
        if grid_state is not None:
            fallback = self._avoid_lava_without_shield(fallback, grid_state, self._legal_actions(grid_state))
        self._dump_frame(state, fallback, [], "fallback_raw", None)
        return fallback

    # -------------------- Core reasoning --------------------

    def _safe_fallback(
        self, grid_state: Optional[GridState], legal_actions: Optional[List[Action]] = None
    ) -> Action:
        """Safe fallback that still tries to move meaningfully."""
        if grid_state is not None:
            legal = legal_actions if legal_actions is not None else self._legal_actions(grid_state)
            if legal:
                if Action.PICK_UP in legal:
                    return Action.PICK_UP
                if Action.USE_KEY in legal:
                    return Action.USE_KEY
                moves = [a for a in legal if a in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)]
                if moves:
                    if self._should_use_blind_explorer():
                        explore_act = self._exploration_action(grid_state, legal)
                        if explore_act is not None:
                            return explore_act
                    best_act = None
                    best_score = 10**9
                    if self._step_input_is_image and self._fast_image_legality:
                        cur = self._find_agent_pos(grid_state)
                        width = int(getattr(grid_state, "width", 0) or 0)
                        height = int(getattr(grid_state, "height", 0) or 0)
                        if cur is not None:
                            for act in moves:
                                npos = self._neighbor_pos(cur, act)
                                if npos is None:
                                    continue
                                nx, ny = npos
                                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                                    continue
                                score = self._visit_counts.get((nx, ny), 0)
                                if score < best_score:
                                    best_score = score
                                    best_act = act
                    else:
                        for act in moves:
                            try:
                                trial_state = copy.deepcopy(grid_state)
                                result = grid_step(trial_state, act)
                                next_state = result[0] if isinstance(result, tuple) else result
                                if next_state is None:
                                    continue
                                npos = self._find_agent_pos(next_state)
                                score = self._visit_counts.get(npos, 0) if npos is not None else 10**8
                                if score < best_score:
                                    best_score = score
                                    best_act = act
                            except Exception:
                                continue
                    if best_act is not None:
                        return best_act
                for _ in range(len(self._dir_cycle)):
                    act = self._dir_cycle[self._dir_idx]
                    self._dir_idx = (self._dir_idx + 1) % len(self._dir_cycle)
                    if act in legal:
                        return act
                return legal[0]
        # No parsed state: avoid permanent WAIT loops.
        act = self._dir_cycle[self._dir_idx]
        self._dir_idx = (self._dir_idx + 1) % len(self._dir_cycle)
        return act

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
            # No turn metadata available; decay conservatively once per step call.
            advanced = True

        if not advanced:
            return

        if self._mem_ghost_turns > 0:
            self._mem_ghost_turns -= 1
        if self._mem_ghost_turns <= 0:
            self._mem_ghost_turns = 0
            self._mem_ghost = False

        if self._exit_memory:
            nxt: Dict[Tuple[int, int], int] = {}
            for pos, ttl in self._exit_memory.items():
                nttl = int(ttl) - 1
                if nttl > 0:
                    nxt[pos] = nttl
            self._exit_memory = nxt

    def _reset_episode_memory(self) -> None:
        self._last_step_agent_pos = None
        self._last_step_action = None
        self._stuck_steps = 0
        self._expected_pickup_pos = None
        self._expected_pickup_kind = None
        self._visit_counts.clear()
        self._tile_label_streaks.clear()
        self._exit_memory.clear()
        self._mem_boots = False
        self._mem_ghost = False
        self._mem_ghost_turns = 0
        self._mem_shield = False
        self._mem_keys = 0

    def _activate_ghost_memory(self) -> None:
        self._mem_ghost = True
        self._mem_ghost_turns = max(self._mem_ghost_turns, self._ghost_turn_duration + 1)

    def _should_use_blind_explorer(self) -> bool:
        c = self._last_label_counts or {}
        visible_goal_like = (
            int(c.get("gem", 0))
            + int(c.get("exit", 0))
            + int(c.get("key", 0))
            + int(c.get("boots", 0))
            + int(c.get("ghost", 0))
            + int(c.get("shield", 0))
            + int(c.get("coin", 0))
        )
        return visible_goal_like == 0

    def _exploration_action(self, grid_state: GridState, legal: List[Action]) -> Optional[Action]:
        moves = [a for a in legal if a in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)]
        if not moves:
            return None
        start = self._find_agent_pos(grid_state)
        if start is None:
            return None
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
        except Exception:
            return None
        if width <= 0 or height <= 0:
            return None

        has_ghost = self._has_ghost_active(grid_state)
        has_shield = self._has_shield_active(grid_state)
        has_key_inv = bool(self._mem_keys > 0)
        walls = set()
        locked_doors = set()
        lava = set()
        keys_on_ground = set()
        exits = set()
        for x in range(width):
            for y in range(height):
                for obj in grid_state.grid[x][y]:
                    n = self._appearance_name(obj)
                    is_agent = getattr(obj, "agent", None) is not None or n in ("agent", "human")
                    if is_agent:
                        try:
                            inv = list(getattr(obj, "inventory_list", None) or [])
                            for item in inv:
                                if getattr(item, "key", None) is not None or self._appearance_name(item) == "key":
                                    has_key_inv = True
                                    break
                        except Exception:
                            pass
                    if getattr(obj, "key", None) is not None or n == "key":
                        keys_on_ground.add((x, y))
                    if getattr(obj, "exit", None) is not None or n == "exit":
                        exits.add((x, y))
                    if getattr(obj, "locked", None) is not None or n in ("locked", "door_locked"):
                        locked_doors.add((x, y))
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

        # Suppress isolated locked-door hallucinations in blind exploration.
        if locked_doors and not (has_key_inv or keys_on_ground or exits):
            locked_doors = set()

        blocked = set()
        if not has_ghost:
            blocked |= walls
            blocked |= locked_doors
        if not has_shield:
            blocked |= lava
        blocked.discard(start)

        dirs = [
            (1, 0, Action.RIGHT),
            (-1, 0, Action.LEFT),
            (0, 1, Action.DOWN),
            (0, -1, Action.UP),
        ]
        q = deque([start])
        seen = {start}
        dist: Dict[Tuple[int, int], int] = {start: 0}
        first_move: Dict[Tuple[int, int], Action] = {}

        while q:
            cx, cy = q.popleft()
            for dx, dy, act in dirs:
                nx, ny = cx + dx, cy + dy
                nxt = (nx, ny)
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if nxt in blocked or nxt in seen:
                    continue
                seen.add(nxt)
                dist[nxt] = dist[(cx, cy)] + 1
                first_move[nxt] = act if (cx, cy) == start else first_move[(cx, cy)]
                q.append(nxt)

        best_target = None
        best_score = None
        for pos in seen:
            if pos == start:
                continue
            score = (self._visit_counts.get(pos, 0), dist.get(pos, 10**9))
            if best_score is None or score < best_score:
                best_score = score
                best_target = pos

        if best_target is None:
            return None
        act = first_move.get(best_target)
        if act is None or act not in moves:
            return None
        return act

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

    def _simulated_next_pos(self, grid_state: GridState, act: Action) -> Optional[Tuple[int, int]]:
        if act not in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            return None
        try:
            trial_state = copy.deepcopy(grid_state)
            result = grid_step(trial_state, act)
            next_state = result[0] if isinstance(result, tuple) else result
            if next_state is None:
                return None
            return self._find_agent_pos(next_state)
        except Exception:
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

    def _pick_escape_move(self, action: Action, grid_state: GridState, legal: List[Action]) -> Optional[Action]:
        moves = (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)
        alts = [alt for alt in moves if alt in legal and alt != action]
        if not alts:
            return None

        cur_pos = self._find_agent_pos(grid_state)
        if cur_pos is None:
            return alts[0]

        # Fast image mode: avoid deep-copy simulation for escape selection.
        if self._step_input_is_image and self._fast_image_legality:
            width = int(getattr(grid_state, "width", 0) or 0)
            height = int(getattr(grid_state, "height", 0) or 0)
            best_alt = None
            best_score = None
            for alt in alts:
                npos = self._neighbor_pos(cur_pos, alt)
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

        best_alt: Optional[Action] = None
        best_score: Optional[Tuple[int, int]] = None
        for alt in alts:
            npos = self._simulated_next_pos(grid_state, alt)
            if npos is None or npos == cur_pos:
                continue
            score = (self._visit_counts.get(npos, 0), 0 if alt != action else 1)
            if best_score is None or score < best_score:
                best_score = score
                best_alt = alt
        if best_alt is not None:
            return best_alt
        return alts[0]

    def _anti_stuck_adjust(self, action: Action, grid_state: GridState, legal: List[Action]) -> Action:
        """Break deterministic loops when perceived position does not change."""
        moves = (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP)

        pos = self._find_agent_pos(grid_state)
        # Immediate guard: don't repeat the same blocked movement direction.
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

        # Original behavior for broader loop breaking.
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
        dxdy = {
            Action.RIGHT: (1, 0),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.UP: (0, -1),
        }
        dx, dy = dxdy[act]
        nx, ny = pos[0] + dx, pos[1] + dy
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

    def _reason_action(self, grid_state: GridState) -> Optional[Action]:
        """Reason directly on grid entities and return a planned action."""
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
                        for item in inv:
                            iname = self._appearance_name(item)
                            if getattr(item, "key", None) is not None or iname == "key":
                                keys_count += 1
                            if getattr(item, "speed", None) is not None or iname == "boots":
                                has_boots = True
                            if getattr(item, "phasing", None) is not None or iname == "ghost":
                                has_ghost = True
                            if getattr(item, "immunity", None) is not None or iname == "shield":
                                has_shield = True
                        for item in status:
                            sname = self._appearance_name(item)
                            if getattr(item, "speed", None) is not None or sname == "boots":
                                has_boots = True
                            if getattr(item, "phasing", None) is not None or sname == "ghost":
                                has_ghost = True
                            if getattr(item, "immunity", None) is not None or sname == "shield":
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

        self._remember_exit_positions(exits)
        exits = self._merge_exit_memory(exits, boxes)

        self._decay_target_memory()
        self._remember_targets("gem", gems)
        self._remember_targets("exit", exits)
        self._remember_targets("key", keys_on_ground)
        self._remember_targets("boots", boots_on_ground)
        self._remember_targets("ghost", ghosts_on_ground)
        self._remember_targets("shield", shields_on_ground)
        self._remember_targets("door_locked", locked_doors)

        gems = self._targets_with_memory("gem", gems)
        exits = self._targets_with_memory("exit", exits)
        keys_on_ground = self._targets_with_memory("key", keys_on_ground)
        boots_on_ground = self._targets_with_memory("boots", boots_on_ground)
        ghosts_on_ground = self._targets_with_memory("ghost", ghosts_on_ground)
        shields_on_ground = self._targets_with_memory("shield", shields_on_ground)
        locked_doors = self._targets_with_memory("door_locked", locked_doors)

        # Ignore isolated locked-door detections unless there is corroborating signal.
        # This prevents hallucinated doors from blocking pathfinding objectives.
        if locked_doors and not (keys_count > 0 or self._mem_keys > 0 or keys_on_ground or exits):
            locked_doors = set()

        # Include memory fallback for powerups that were picked but not reflected in parsed status.
        has_boots = has_boots or self._mem_boots
        has_ghost = has_ghost or self._mem_ghost or self._mem_ghost_turns > 0
        has_shield = has_shield or self._mem_shield
        keys_count += max(0, int(self._mem_keys))

        # Interactions at current tile first.
        if (
            agent_pos in gems
            or agent_pos in coins
            or agent_pos in keys_on_ground
            or agent_pos in boots_on_ground
            or agent_pos in ghosts_on_ground
            or agent_pos in shields_on_ground
        ):
            if agent_pos in boots_on_ground:
                self._mem_boots = True
            if agent_pos in ghosts_on_ground:
                self._activate_ghost_memory()
            if agent_pos in shields_on_ground:
                self._mem_shield = True
            if agent_pos in keys_on_ground:
                self._mem_keys += 1
            self._forget_target_at(agent_pos)
            self._expected_pickup_pos = None
            self._expected_pickup_kind = None
            return Action.PICK_UP
        if keys_count > 0:
            ax, ay = agent_pos
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if (ax + dx, ay + dy) in locked_doors:
                    if self._mem_keys > 0:
                        self._mem_keys -= 1
                    self._forget_target_at((ax + dx, ay + dy))
                    return Action.USE_KEY

        # If we already have a key, explicitly navigate to a locked door to open it.
        door_adj_targets = set()
        if keys_count > 0 and locked_doors:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                for lx, ly in locked_doors:
                    tx, ty = lx + dx, ly + dy
                    if tx < 0 or ty < 0 or tx >= width or ty >= height:
                        continue
                    if (tx, ty) in walls:
                        continue
                    if not has_shield and (tx, ty) in lava:
                        continue
                    door_adj_targets.add((tx, ty))

        # Set objective priority with explicit mechanics constraints:
        # 1) If there are locked doors and we don't have a key, get a key first.
        # 2) If lava exists and shield is not active, get shield first.
        targets = []
        powerups = list(shields_on_ground | boots_on_ground | ghosts_on_ground)
        shield_targets = list(shields_on_ground)
        need_shield_for_lava = bool(lava) and not has_shield
        need_key_for_door = bool(locked_doors) and keys_count == 0 and bool(keys_on_ground)

        if need_key_for_door:
            targets = list(keys_on_ground)
        elif door_adj_targets:
            targets = list(door_adj_targets)
        elif need_shield_for_lava and shield_targets:
            targets = shield_targets
        elif gems:
            targets = list(gems)
        elif exits:
            targets = list(exits)
        elif keys_on_ground:
            targets = list(keys_on_ground)
        elif powerups:
            targets = powerups
        elif coins:
            targets = list(coins)

        if not targets:
            self._expected_pickup_pos = None
            self._expected_pickup_kind = None
            return None

        # While ghost is active, immediately exploit phasing if that unlocks objectives.
        if has_ghost and self._mem_ghost_turns > 0:
            blocked_no_phase = set(walls) | set(locked_doors)
            if not has_shield:
                blocked_no_phase |= lava
            blocked_no_phase.discard(agent_pos)

            blocked_with_phase = set()
            if not has_shield:
                blocked_with_phase |= lava
            blocked_with_phase.discard(agent_pos)

            ghost_targets = []
            if gems:
                ghost_targets = list(gems)
            elif exits:
                ghost_targets = list(exits)
            elif door_adj_targets:
                ghost_targets = list(door_adj_targets)
            else:
                ghost_targets = list(targets)

            plain_action, _ = self._bfs_first_action(agent_pos, ghost_targets, width, height, blocked_no_phase)
            phase_action, phase_target = self._bfs_first_action(agent_pos, ghost_targets, width, height, blocked_with_phase)
            if phase_action is not None and plain_action is None:
                self._set_expected_pickup(phase_target, keys_on_ground, shields_on_ground, boots_on_ground, ghosts_on_ground)
                return phase_action

        blocked = set()
        if not has_ghost:
            blocked |= walls
            blocked |= locked_doors
        if not has_shield:
            blocked |= lava
        blocked.discard(agent_pos)

        action, found_target = self._bfs_first_action(agent_pos, targets, width, height, blocked)
        # If main objective is unreachable, try obtaining a powerup that may unlock traversal.
        if action is None and need_key_for_door and keys_on_ground:
            action, found_target = self._bfs_first_action(agent_pos, list(keys_on_ground), width, height, blocked)
        if action is None and need_shield_for_lava and shield_targets:
            action, found_target = self._bfs_first_action(agent_pos, shield_targets, width, height, blocked)
        if action is None and powerups and not (has_ghost and has_boots and has_shield):
            action, found_target = self._bfs_first_action(agent_pos, powerups, width, height, blocked)
        self._set_expected_pickup(found_target, keys_on_ground, shields_on_ground, boots_on_ground, ghosts_on_ground)
        return action

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

    def _remember_exit_positions(self, exits: set) -> None:
        for pos in exits:
            self._exit_memory[pos] = self._exit_memory_ttl

    def _merge_exit_memory(self, observed_exits: set, boxes: set) -> set:
        out = set(observed_exits)
        for pos, ttl in self._exit_memory.items():
            if ttl <= 0:
                continue
            # Keep remembered exits while they may be hidden by a pushable box.
            if pos in boxes or pos not in observed_exits:
                out.add(pos)
        return out

    def _decay_target_memory(self) -> None:
        if self._target_memory_ttl <= 0:
            return
        for kind in list(self._target_memory.keys()):
            cur = self._target_memory.get(kind, {})
            nxt: Dict[Tuple[int, int], int] = {}
            for pos, ttl in cur.items():
                nttl = int(ttl) - 1
                if nttl > 0:
                    nxt[pos] = nttl
            self._target_memory[kind] = nxt

    def _remember_targets(self, kind: str, positions: set) -> None:
        if self._target_memory_ttl <= 0:
            return
        mem = self._target_memory.setdefault(kind, {})
        for pos in positions:
            mem[pos] = self._target_memory_ttl

    def _targets_with_memory(self, kind: str, observed: set) -> set:
        if self._target_memory_ttl <= 0:
            return set(observed)
        mem = self._target_memory.get(kind, {})
        if not mem:
            return set(observed)
        out = set(observed)
        out.update(mem.keys())
        return out

    def _forget_target_at(self, pos: Tuple[int, int]) -> None:
        for kind in self._target_memory.keys():
            self._target_memory[kind].pop(pos, None)

    def _apply_expected_pickup_memory(self) -> None:
        if self._expected_pickup_kind == "key":
            self._mem_keys += 1
        elif self._expected_pickup_kind == "shield":
            self._mem_shield = True
        elif self._expected_pickup_kind == "boots":
            self._mem_boots = True
        elif self._expected_pickup_kind == "ghost":
            self._activate_ghost_memory()

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

    def _state_fingerprint(self, grid_state: GridState) -> Tuple:
        """Compact fingerprint for detecting no-op transitions."""
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
        except Exception:
            return ()
        agent_pos = self._find_agent_pos(grid_state)
        counts = {
            "gem": 0,
            "coin": 0,
            "key": 0,
            "locked": 0,
            "exit": 0,
        }
        for x in range(width):
            for y in range(height):
                for obj in grid_state.grid[x][y]:
                    n = self._appearance_name(obj)
                    if getattr(obj, "requirable", None) is not None or n in ("gem", "core"):
                        counts["gem"] += 1
                    if getattr(obj, "rewardable", None) is not None and n == "coin":
                        counts["coin"] += 1
                    if getattr(obj, "key", None) is not None or n == "key":
                        counts["key"] += 1
                    if getattr(obj, "locked", None) is not None or n in ("locked", "door_locked"):
                        counts["locked"] += 1
                    if getattr(obj, "exit", None) is not None or n == "exit":
                        counts["exit"] += 1
        return (
            agent_pos,
            counts["gem"],
            counts["coin"],
            counts["key"],
            counts["locked"],
            counts["exit"],
        )

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
            # Check inventory on the agent tile.
            for obj in grid_state.grid[ax][ay]:
                is_agent = getattr(obj, "agent", None) is not None or self._appearance_name(obj) in (
                    "agent",
                    "human",
                )
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
            # Need adjacent locked door.
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

    def _can_attempt_move_with_memory(self, grid_state: GridState, act: Action) -> bool:
        if act not in (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP):
            return False
        pos = self._find_agent_pos(grid_state)
        if pos is None:
            return False
        dxdy = {
            Action.RIGHT: (1, 0),
            Action.DOWN: (0, 1),
            Action.LEFT: (-1, 0),
            Action.UP: (0, -1),
        }
        dx, dy = dxdy[act]
        x, y = pos
        nx, ny = x + dx, y + dy
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                return False
            # Only relax legality if destination has something that memory could bypass.
            for obj in grid_state.grid[nx][ny]:
                n = self._appearance_name(obj)
                if (getattr(obj, "locked", None) is not None or n in ("locked", "door_locked")) and (
                    self._mem_ghost or self._mem_ghost_turns > 0
                ):
                    return True
                if (getattr(obj, "damage", None) is not None or n == "lava") and self._mem_shield:
                    return True
                is_wall = n == "wall" or (
                    getattr(obj, "blocking", None) is not None
                    and getattr(obj, "locked", None) is None
                    and getattr(obj, "pushable", None) is None
                    and n not in ("door", "opened", "door_open")
                )
                if is_wall and (self._mem_ghost or self._mem_ghost_turns > 0):
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
            has_shield = self._has_shield_active(grid_state)
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
                is_lava = getattr(obj, "damage", None) is not None or n == "lava"
                if is_lava and not has_shield:
                    # Still allow as candidate; separate lava-avoidance logic handles preference.
                    continue
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

    def _legal_actions_fast(self, grid_state: GridState) -> List[Action]:
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

    def _legal_actions(self, grid_state: GridState) -> List[Action]:
        if self._step_input_is_image and self._fast_image_legality:
            return self._legal_actions_fast(grid_state)
        legal: List[Action] = []
        for act in [
            Action.PICK_UP,
            Action.USE_KEY,
            Action.RIGHT,
            Action.DOWN,
            Action.LEFT,
            Action.UP,
        ]:
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
                elif self._can_attempt_move_with_memory(grid_state, act):
                    legal.append(act)
            except Exception:
                continue
        if not legal:
            legal.append(Action.WAIT)
        return legal

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

    def _action_name(self, action) -> str:
        if isinstance(action, Action):
            return action.name
        return str(action)

    def _extract_level_hint(self, info) -> Dict[str, object]:
        out: Dict[str, object] = {}
        if not isinstance(info, dict):
            return out
        for k in ("level", "level_name", "level_id", "builder", "builder_name", "scenario", "stage"):
            if k in info:
                out[k] = info.get(k)
        cfg = info.get("config")
        if isinstance(cfg, dict):
            for k in ("movement", "objective", "seed", "width", "height", "turn_limit"):
                if k in cfg:
                    out[f"config_{k}"] = cfg.get(k)
        return out

    def _cell_primary_label(self, grid_state: GridState, x: int, y: int) -> str:
        try:
            objs = list(grid_state.grid[x][y])
        except Exception:
            return "floor"

        has_blocking_wall = False
        has_box = False
        has_lava = False
        has_exit = False
        has_locked = False
        has_open_door = False
        has_gem = False
        has_coin = False
        has_key = False
        has_boots = False
        has_ghost = False
        has_shield = False
        has_agent = False

        for obj in objs:
            n = self._appearance_name(obj)
            if getattr(obj, "agent", None) is not None or n in ("agent", "human"):
                has_agent = True
            if getattr(obj, "requirable", None) is not None or n in ("gem", "core"):
                has_gem = True
            if getattr(obj, "rewardable", None) is not None and n == "coin":
                has_coin = True
            if getattr(obj, "key", None) is not None or n == "key":
                has_key = True
            if getattr(obj, "speed", None) is not None or n == "boots":
                has_boots = True
            if getattr(obj, "phasing", None) is not None or n == "ghost":
                has_ghost = True
            if getattr(obj, "immunity", None) is not None or n == "shield":
                has_shield = True
            if getattr(obj, "locked", None) is not None or n in ("locked", "door_locked"):
                has_locked = True
            if n in ("opened", "door_open"):
                has_open_door = True
            if getattr(obj, "exit", None) is not None or n == "exit":
                has_exit = True
            if getattr(obj, "damage", None) is not None or n == "lava":
                has_lava = True
            if getattr(obj, "pushable", None) is not None or n == "box":
                has_box = True
            is_wall = n == "wall" or (
                getattr(obj, "blocking", None) is not None
                and getattr(obj, "locked", None) is None
                and getattr(obj, "pushable", None) is None
                and n not in ("door", "opened", "door_open")
            )
            if is_wall:
                has_blocking_wall = True

        if has_agent:
            return "agent"
        if has_gem:
            return "gem"
        if has_coin:
            return "coin"
        if has_key:
            return "key"
        if has_boots:
            return "boots"
        if has_ghost:
            return "ghost"
        if has_shield:
            return "shield"
        if has_exit:
            return "exit"
        if has_locked:
            return "door_locked"
        if has_open_door:
            return "door_open"
        if has_box:
            return "box"
        if has_lava:
            return "lava"
        if has_blocking_wall:
            return "wall"
        return "floor"

    def _grid_label_map(self, grid_state: Optional[GridState]) -> List[List[str]]:
        if grid_state is None:
            return []
        try:
            width = int(getattr(grid_state, "width", 0))
            height = int(getattr(grid_state, "height", 0))
            if width <= 0 or height <= 0:
                return []
        except Exception:
            return []

        rows: List[List[str]] = []
        for y in range(height):
            row: List[str] = []
            for x in range(width):
                row.append(self._cell_primary_label(grid_state, x, y))
            rows.append(row)
        return rows

    def _dump_frame(
        self,
        state: GridState | ImageObservation,
        chosen: Action,
        legal: List[Action],
        source: str,
        grid_state: Optional[GridState],
    ) -> None:
        if not self._debug:
            return
        try:
            image, info = self._extract_observation(state)
            turn = self._extract_turn(info)
            if turn is None and isinstance(state, GridState):
                maybe_turn = getattr(state, "turn", None)
                try:
                    turn = int(maybe_turn) if maybe_turn is not None else None
                except Exception:
                    turn = None
            self._debug_seq += 1
            t = f"{turn:04d}" if turn is not None else "none"
            tag = f"run_{self._debug_run_id}_seq_{self._debug_seq:06d}_turn_{t}"
            metadata = {
                "tag": tag,
                "source": source,
                "chosen": self._action_name(chosen),
                "legal": [self._action_name(a) for a in legal],
                "hint_agent_pos": info.get("agent_pos") if isinstance(info, dict) else None,
                "parsed_agent_pos": self._last_agent_pos,
                "flip_y": self._last_flip_y,
                "label_counts": self._last_label_counts,
                "level_hint": self._extract_level_hint(info),
                "ml_local_conf": self._debug_local_ml_conf(image, grid_state),
                "target_memory_sizes": {k: len(v) for k, v in self._target_memory.items()},
            }
            if grid_state is not None:
                metadata["grid_width"] = int(getattr(grid_state, "width", 0))
                metadata["grid_height"] = int(getattr(grid_state, "height", 0))
                metadata["label_grid"] = self._grid_label_map(grid_state)
            elif self._last_label_grid:
                metadata["label_grid"] = self._last_label_grid

            import json

            meta_path = os.path.join(self._debug_dir, f"{tag}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)

            if image is None or np is None:
                return
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return
            try:
                from PIL import Image

                rgb = arr[..., :3].astype(np.uint8)
                img_path = os.path.join(self._debug_dir, f"{tag}.png")
                Image.fromarray(rgb).save(img_path)
            except Exception:
                return
        except Exception:
            return

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

        grid_w, grid_h = self._infer_grid_size(info, img.shape[1], img.shape[0], img)
        if not grid_w or not grid_h:
            return self._reusable_last_state(info)

        tile_w = img.shape[1] // grid_w
        tile_h = img.shape[0] // grid_h
        if tile_w <= 0 or tile_h <= 0:
            return self._reusable_last_state(info)

        state = self._make_empty_gridstate(grid_w, grid_h)
        if state is None:
            return self._reusable_last_state(info)

        # Import entities lazily.
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
                label = self._temporal_smooth_label(x, y, label)
                if label is None or label == "floor":
                    continue
                recognized.append((x, y, label))
                label_counts[label] = label_counts.get(label, 0) + 1

        # Recover a common confusion: right-side exit predicted as a single locked door.
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

        label_grid = [["floor" for _ in range(grid_w)] for _ in range(grid_h)]
        for x, y, label in recognized:
            if label == "agent" and chosen_agent_raw is not None and (x, y) != chosen_agent_raw:
                continue
            ent_cls = entity_by_label.get(label)
            if ent_cls is None:
                continue
            game_pos = self._to_grid_pos(x, y, grid_h, flip_y)
            state.add(game_pos, ent_cls())
            gx, gy = game_pos
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                label_grid[gy][gx] = label
            if label == "agent":
                agent_pos = game_pos

        # Agent recovery from info / memory if missed.
        if agent_pos is None:
            hinted = self._hinted_agent_pos(info, grid_w, grid_h)
            if hinted is not None:
                agent_pos = hinted
                state.add(agent_pos, entity_by_label["agent"]())
                hx, hy = agent_pos
                if 0 <= hx < grid_w and 0 <= hy < grid_h:
                    label_grid[hy][hx] = "agent"
        self._last_label_grid = label_grid

        if agent_pos is not None:
            if not self._is_plausible_parse(grid_w, grid_h, label_counts, state):
                reused = self._reusable_last_state(info)
                if reused is not None:
                    return reused
                # Reject implausible parses instead of committing stale state.
                self._consecutive_reuse = 0
                return None
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
        # Very loose sanity checks that reject only obviously corrupted parses.
        cells = max(1, grid_w * grid_h)
        agent_n = int(label_counts.get("agent", 0))
        if agent_n > 2:
            return False
        # In noisy frames the agent sprite can be missed; allow this if we still
        # reconstructed an agent via hint/memory.
        if agent_n == 0 and self._find_agent_pos(state) is None:
            return False
        # Doors/powerups/boxes are usually sparse in intro levels; large counts indicate template drift.
        if label_counts.get("door_locked", 0) > max(3, cells // 5):
            return False
        if label_counts.get("box", 0) > max(4, cells // 4):
            return False
        if label_counts.get("shield", 0) + label_counts.get("ghost", 0) + label_counts.get("boots", 0) > max(4, cells // 4):
            return False
        # If almost everything is wall/lava, parse likely collapsed.
        blocked_like = int(label_counts.get("wall", 0)) + int(label_counts.get("lava", 0))
        if blocked_like > int(cells * 0.85):
            return False
        # Ensure an agent exists in reconstructed state.
        if self._find_agent_pos(state) is None:
            return False
        return True

    # -------------------- Grid size inference --------------------

    def _infer_grid_size(
        self, info, width_px: int, height_px: int, img: Optional["np.ndarray"] = None
    ) -> Tuple[Optional[int], Optional[int]]:
        # 1) Explicit metadata first.
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

        # 2) Candidate search.
        g = math.gcd(width_px, height_px)
        candidates: List[Tuple[int, int, int]] = []  # (gw, gh, tile)
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

        # Fixed tile renderers are common (64px). Prefer this deterministic path
        # to avoid expensive candidate scoring on first frame.
        for gw, gh, tile in candidates:
            if tile == 64:
                self._grid_size_cache[key] = (gw, gh)
                return gw, gh

        if img is None:
            # Prefer practical renderer tile sizes (64 is common at 640 resolution).
            best = min(candidates, key=lambda t: abs(t[2] - 64))
            self._grid_size_cache[key] = (best[0], best[1])
            return best[0], best[1]

        # 3) Score candidates by recognition confidence on sampled tiles.
        # Keep this bounded; template-bank scoring across many tile sizes is costly.
        candidates = sorted(candidates, key=lambda t: (abs(t[2] - 64), abs(t[0] - t[1])))
        if len(candidates) > 6:
            best = candidates[0]
            self._grid_size_cache[key] = (best[0], best[1])
            return best[0], best[1]
        candidates = candidates[:4]

        best_pair = None
        best_score = -1e9
        for gw, gh, tile in candidates:
            s = self._score_grid_candidate(img, gw, gh, tile)
            if s > best_score:
                best_score = s
                best_pair = (gw, gh)
        if best_pair is None:
            return None, None
        self._grid_size_cache[key] = best_pair
        return best_pair

    def _score_grid_candidate(self, img: "np.ndarray", gw: int, gh: int, tile: int) -> float:
        tile_w = tile
        tile_h = tile
        xs = np.linspace(0, gw - 1, min(gw, 6), dtype=int)
        ys = np.linspace(0, gh - 1, min(gh, 6), dtype=int)
        total = 0
        recognized = 0
        non_floor = 0
        agent_hits = 0
        for yy in ys:
            for xx in xs:
                patch = img[yy * tile_h : (yy + 1) * tile_h, xx * tile_w : (xx + 1) * tile_w]
                label = self._classify_tile(patch, tile_h, tile_w)
                if label is None:
                    label = self._heuristic_tile(patch)
                total += 1
                if label is not None:
                    recognized += 1
                    if label != "floor":
                        non_floor += 1
                    if label == "agent":
                        agent_hits += 1
        if total == 0:
            return -1e9
        score = (recognized / total) + 0.4 * (non_floor / total)
        if agent_hits == 1:
            score += 0.5
        elif agent_hits > 1:
            score -= 0.3
        score -= abs(tile - 64) / 256.0
        return score

    # -------------------- Template classifier --------------------

    def _load_assets(self) -> Dict[str, List["np.ndarray"]]:
        if np is None:
            return {}
        assets: Dict[str, List["np.ndarray"]] = {}
        roots = [os.path.join(os.getcwd(), "data", "assets")]

        file_path = globals().get("__file__")
        if isinstance(file_path, str) and file_path:
            roots.append(os.path.join(os.path.dirname(os.path.abspath(file_path)), "data", "assets"))
        # Coursemology packaging may not include local data/assets.
        # Fallback to sprites bundled with the installed grid_adventure package.
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

        # Resource API fallback (works when package data is not exposed as real files).
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
            try:
                import matplotlib.image as mpimg

                arr = mpimg.imread(path)
                if arr is None:
                    return None
                arr = np.asarray(arr)
                if arr.ndim != 3:
                    return None
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
                if arr.shape[2] == 3:
                    alpha = np.full((*arr.shape[:2], 1), 255, dtype=arr.dtype)
                    arr = np.concatenate([arr, alpha], axis=2)
                elif arr.shape[2] > 4:
                    arr = arr[..., :4]
                return arr
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
            try:
                import matplotlib.image as mpimg
                import io

                arr = mpimg.imread(io.BytesIO(data))
                if arr is None:
                    return None
                arr = np.asarray(arr)
                if arr.ndim != 3:
                    return None
                if arr.dtype != np.uint8:
                    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
                if arr.shape[2] == 3:
                    alpha = np.full((*arr.shape[:2], 1), 255, dtype=arr.dtype)
                    arr = np.concatenate([arr, alpha], axis=2)
                elif arr.shape[2] > 4:
                    arr = arr[..., :4]
                return arr
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
            ml_out = self._classify_tile_ml(tile_img)
            self._tile_cache[key] = ml_out
            return ml_out

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

        # Exit can appear visually close to floor in some seeds; recover this tie-case.
        if out == "floor":
            floor_score = label_scores.get("floor")
            exit_score = label_scores.get("exit")
            if floor_score is not None and exit_score is not None:
                if floor_score >= 0.020 and exit_score <= floor_score * 1.16 and exit_score <= 0.050:
                    out = "exit"

        ml_label, ml_conf = self._classify_tile_ml_with_conf(tile_img)
        critical = {"key", "door_open", "shield", "ghost", "boots", "gem"}

        # If ML is very confident on mechanics-critical classes, let it override template.
        if ml_label in critical and ml_conf is not None:
            need = self._ml_class_thresholds.get(ml_label, self._ml_confidence_threshold)
            if ml_conf >= need + 0.08:
                out = ml_label

        # Otherwise keep template-first. Avoid ML fallback for background classes,
        # so heuristics can still recover exits when template matching is ambiguous.
        if out is None:
            if ml_label not in ("floor", "wall", "lava"):
                out = ml_label
        self._tile_cache[key] = out
        return out

    def _classify_tile_ml(self, tile_img: "np.ndarray") -> Optional[str]:
        out, _ = self._classify_tile_ml_with_conf(tile_img)
        return out

    def _classify_tile_ml_with_conf(self, tile_img: "np.ndarray") -> Tuple[Optional[str], Optional[float]]:
        if np is None or torch is None or self._ml_model is None or not self._ml_labels:
            return None, None
        arr = tile_img
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None, None
        try:
            from PIL import Image

            rgb = arr[..., :3].astype(np.uint8)
            im = Image.fromarray(rgb).resize((self._ml_image_size, self._ml_image_size), resample=Image.NEAREST)
            x = np.asarray(im, dtype=np.float32) / 255.0
            ten = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                logits = self._ml_model(ten)
                probs = torch.softmax(logits, dim=1)
                conf, idx = torch.max(probs, dim=1)
            c = float(conf.item())
            i = int(idx.item())
            if i < 0 or i >= len(self._ml_labels):
                return None, None
            label = self._ml_labels[i]
            min_conf = self._ml_class_thresholds.get(label, self._ml_confidence_threshold)
            if c < min_conf:
                return None, c
            return label, c
        except Exception:
            return None, None

    def _debug_local_ml_conf(self, image, grid_state: Optional[GridState]) -> List[Dict]:
        if not self._debug:
            return []
        if np is None or grid_state is None:
            return []
        arr = np.asarray(image) if image is not None else None
        if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
            return []
        try:
            gw = int(getattr(grid_state, "width", 0))
            gh = int(getattr(grid_state, "height", 0))
            if gw <= 0 or gh <= 0:
                return []
            tw = arr.shape[1] // gw
            th = arr.shape[0] // gh
            if tw <= 0 or th <= 0:
                return []
            pos = self._find_agent_pos(grid_state) or self._last_agent_pos
            if pos is None:
                return []
            ax, ay = pos
            out: List[Dict] = []
            for gx, gy in ((ax, ay), (ax + 1, ay), (ax - 1, ay), (ax, ay + 1), (ax, ay - 1)):
                if gx < 0 or gy < 0 or gx >= gw or gy >= gh:
                    continue
                tile = arr[gy * th : (gy + 1) * th, gx * tw : (gx + 1) * tw]
                label, conf = self._classify_tile_ml_with_conf(tile)
                if conf is None:
                    continue
                out.append(
                    {
                        "pos": [int(gx), int(gy)],
                        "label": label,
                        "conf": round(float(conf), 4),
                    }
                )
            out.sort(key=lambda d: d.get("conf", 0.0), reverse=True)
            return out[:3]
        except Exception:
            return []

    def _temporal_smooth_label(self, x: int, y: int, label: Optional[str]) -> Optional[str]:
        pos = (x, y)
        if label is None:
            self._tile_label_streaks.pop(pos, None)
            return None
        if label not in self._temporal_unstable_labels:
            self._tile_label_streaks[pos] = (label, 1)
            return label

        prev = self._tile_label_streaks.get(pos)
        if prev is not None and prev[0] == label:
            streak = prev[1] + 1
        else:
            streak = 1
        self._tile_label_streaks[pos] = (label, streak)
        if streak >= self._temporal_min_streak:
            return label
        return None

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
