# AgentImage.py
from __future__ import annotations

import os
from collections import deque
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:  # numpy is expected in the grader environment
    np = None

from grid_adventure.step import Action

# Attempt to reuse the Task 1 planner logic from Agent.py (A* pathfinding agent).
try:
    from Agent import Agent as GridAgent
except Exception:
    GridAgent = None
    try:
        import importlib.util
        _agent_path = os.path.join(os.getcwd(), "Agent.py")
        if os.path.isfile(_agent_path):
            _spec = importlib.util.spec_from_file_location("_agent_fallback", _agent_path)
            if _spec and _spec.loader:
                _mod = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                GridAgent = getattr(_mod, "Agent", None)
    except Exception:
        GridAgent = None


class _FallbackPlanner:
    """Minimal planner used only if Agent.py cannot be loaded."""
    def step(self, _state) -> Action:
        return Action.WAIT


class _ProtoTileClassifier:
    """
    Fast tile classifier: nearest centroid ("prototype") in a small RGBA feature space.

    Model file: .npz with keys:
      - labels: object array of strings, shape [C]
      - centroids: float32 array, shape [C,D]
      - thresholds: float32 array, shape [C]  (class accept threshold)
      - feat_hw: int (feature size H=W)
    """
    def __init__(self, model_path: Optional[str] = None):
        self.labels: List[str] = []
        self.centroids = None           # [C,D] float32
        self.thresholds = None          # [C] float32
        self.feat_hw: int = 16

        # precomputed for fast distance
        self._centroids_norm2 = None    # [C]
        if model_path:
            self.load(model_path)

    def is_ready(self) -> bool:
        return (np is not None and
                self.centroids is not None and
                self.thresholds is not None and
                len(self.labels) == int(self.centroids.shape[0]))

    def load(self, path: str) -> bool:
        if np is None:
            return False
        if not path or not os.path.isfile(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)
            labels = data.get("labels")
            centroids = data.get("centroids")
            thresholds = data.get("thresholds")
            feat_hw = data.get("feat_hw", 16)

            if labels is None or centroids is None or thresholds is None:
                return False

            self.labels = [str(x) for x in list(labels)]
            self.centroids = centroids.astype(np.float32, copy=False)
            self.thresholds = thresholds.astype(np.float32, copy=False)
            self.feat_hw = int(feat_hw)
            self._centroids_norm2 = np.sum(self.centroids * self.centroids, axis=1)
            return True
        except Exception:
            return False

    def _resize_nn_rgba(self, img: "np.ndarray", w: int, h: int) -> "np.ndarray":
        # Prefer PIL if present, else manual
        try:
            from PIL import Image
            im = Image.fromarray(img)
            im = im.resize((w, h), resample=Image.NEAREST)
            return np.asarray(im)
        except Exception:
            y_idx = (np.linspace(0, img.shape[0] - 1, h)).astype(int)
            x_idx = (np.linspace(0, img.shape[1] - 1, w)).astype(int)
            return img[y_idx][:, x_idx]

    def featurize(self, tile_img: "np.ndarray") -> "np.ndarray":
        # ensure RGBA
        if tile_img.ndim != 3:
            raise ValueError("tile_img must be HxWxC")
        if tile_img.shape[2] == 3:
            alpha = np.full((*tile_img.shape[:2], 1), 255, dtype=tile_img.dtype)
            tile_img = np.concatenate([tile_img, alpha], axis=2)
        elif tile_img.shape[2] >= 4:
            tile_img = tile_img[..., :4]

        small = self._resize_nn_rgba(tile_img, self.feat_hw, self.feat_hw)
        x = small.astype(np.float32) / 255.0
        return x.reshape(-1)  # [D]

    def predict(self, tile_img: "np.ndarray", accept_margin: float = 1.25) -> Optional[str]:
        """
        Return a label if confident; otherwise None.
        accept_margin: allow some slack beyond the class threshold.
        """
        if not self.is_ready():
            return None

        x = self.featurize(tile_img)  # [D]
        x2 = float(np.dot(x, x))
        # dist^2 = ||c||^2 + ||x||^2 - 2 c·x
        dots = self.centroids @ x  # [C]
        dist = (self._centroids_norm2 + x2 - 2.0 * dots) / float(x.size)  # MSE-like

        idx = int(np.argmin(dist))
        best_d = float(dist[idx])
        thr = float(self.thresholds[idx]) * float(accept_margin) + 1e-8
        if best_d <= thr:
            return self.labels[idx]
        return None


class Agent:
    """Grid Adventure V1 AI Agent (Task 2: ImageObservation).

    Parses RGBA image observations to reconstruct the GridState, then uses a planner
    (from Task 1 or fallback) to decide actions.
    """
    def __init__(self):
        self._planner = GridAgent() if GridAgent is not None else _FallbackPlanner()

        self._assets_loaded = False
        self._assets: Dict[str, List["np.ndarray"]] = {}
        self._template_cache: Dict[Tuple[int, int], dict] = {}
        self._tile_cache: Dict[bytes, Optional[str]] = {}
        self._last_tile_size: Optional[Tuple[int, int]] = None
        self._last_items: Dict[Tuple[int, int], str] = {}

        # Prototype classifier (trained from data/)
        self._proto = None
        if np is not None:
            model_path = os.getenv("GRID_ADVENTURE_TILE_MODEL", "")
            candidates = [
                model_path,
                os.path.join(os.getcwd(), "data", "tile_prototypes.npz"),
                os.path.join(os.getcwd(), "tile_prototypes.npz"),
            ]
            for p in candidates:
                if p and os.path.isfile(p):
                    clf = _ProtoTileClassifier(p)
                    if clf.is_ready():
                        self._proto = clf
                        break

    def step(self, state) -> Action:
        if self._is_image_observation(state):
            parsed_state = self.parse(state)
            if parsed_state is None:
                return Action.WAIT
            return self._planner.step(parsed_state)
        return self._planner.step(state)

    def parse(self, observation):
        image, info = self._extract_observation(observation)

        if np is None:
            grid_w, grid_h = self._infer_grid_size(info, 0, 0)
            if grid_w is None or grid_h is None:
                grid_w, grid_h = 1, 1
            return self._build_empty_gridstate(grid_w, grid_h)

        # reset per-episode memory
        if isinstance(info, dict) and info.get("turn") == 0:
            self._last_items.clear()
            if hasattr(self._planner, "plan"):
                self._planner.plan = deque()
            if hasattr(self._planner, "last_plan_goal"):
                self._planner.last_plan_goal = None

        try:
            from grid_adventure.grid import GridState
        except Exception:
            grid_w, grid_h = self._infer_grid_size(info, 0, 0)
            return self._build_empty_gridstate(max(1, grid_w or 1), max(1, grid_h or 1))

        if isinstance(observation, GridState):
            return observation
        if isinstance(info, dict):
            maybe_grid = info.get("gridstate")
            if isinstance(maybe_grid, GridState):
                return maybe_grid

        if image is None:
            grid_w, grid_h = self._infer_grid_size(info, 0, 0)
            if grid_w is None or grid_h is None:
                grid_w, grid_h = 1, 1
            return self._build_empty_gridstate(grid_w, grid_h)

        img = np.asarray(image)
        if img.ndim != 3 or img.shape[2] < 3:
            grid_w, grid_h = self._infer_grid_size(info, 0, 0)
            if grid_w is None or grid_h is None:
                grid_w, grid_h = 1, 1
            return self._build_empty_gridstate(grid_w, grid_h)

        grid_w, grid_h = self._infer_grid_size(info, img.shape[1], img.shape[0])
        if grid_w is None or grid_h is None:
            return self._build_empty_gridstate(1, 1)

        tile_w = img.shape[1] // grid_w
        tile_h = img.shape[0] // grid_h
        if tile_w <= 0 or tile_h <= 0:
            return self._build_empty_gridstate(max(1, grid_w), max(1, grid_h))

        # clear cache when tile size changes
        if self._last_tile_size != (tile_h, tile_w):
            self._tile_cache.clear()
            self._last_tile_size = (tile_h, tile_w)

        # Only prepare expensive templates if we DON'T have the prototype classifier
        templates = None
        if self._proto is None:
            templates = self._get_templates(tile_h, tile_w)

        # import entities
        try:
            from grid_adventure.entities import (
                AgentEntity, ExitEntity, GemEntity, CoinEntity, LockedDoorEntity,
                BoxEntity, KeyEntity, SpeedPowerUpEntity, ShieldPowerUpEntity, PhasingPowerUpEntity,
                FloorEntity, WallEntity, LavaEntity
            )
            try:
                from grid_adventure.entities import UnlockedDoorEntity
            except Exception:
                UnlockedDoorEntity = None
        except Exception:
            return self._build_empty_gridstate(max(1, grid_w), max(1, grid_h))

        # init GridState
        try:
            from grid_adventure.movements import MOVEMENTS
            from grid_adventure.objectives import OBJECTIVES
            movement = MOVEMENTS.get("cardinal") or MOVEMENTS.get("default")
            objective = OBJECTIVES.get("collect_gems_and_exit") or OBJECTIVES.get("default")
            state = GridState(width=grid_w, height=grid_h, movement=movement, objective=objective)
        except Exception:
            state = GridState(width=grid_w, height=grid_h)

        # fill with floor (note: y flip)
        for x in range(grid_w):
            for y in range(grid_h):
                state.add((x, grid_h - 1 - y), FloorEntity())

        # map appearance keywords -> entity classes (add dataset aliases)
        entity_map: Dict[str, type] = {
            "agent": AgentEntity,
            "human": AgentEntity,

            "exit": ExitEntity,

            "gem": GemEntity,
            "core": GemEntity,
            "floorgem": GemEntity,
            "floor_gem": GemEntity,

            "coin": CoinEntity,

            "key": KeyEntity,

            "door": LockedDoorEntity,
            "locked": LockedDoorEntity,

            "box": BoxEntity,

            "wall": WallEntity,
            "lava": LavaEntity,

            "boots": SpeedPowerUpEntity,
            "bots": SpeedPowerUpEntity,
            "speed": SpeedPowerUpEntity,

            "shield": ShieldPowerUpEntity,

            "ghost": PhasingPowerUpEntity,
            "phasing": PhasingPowerUpEntity,
        }
        if UnlockedDoorEntity is not None:
            entity_map.update({
                "door_open": UnlockedDoorEntity,
                "door_unlocked": UnlockedDoorEntity,
                "unlocked_door": UnlockedDoorEntity,
                "opened": UnlockedDoorEntity,
                "open": UnlockedDoorEntity,
            })

        def to_entity_class(name: str) -> Optional[type]:
            name_l = name.lower()
            for key, cls in entity_map.items():
                if key in name_l:
                    return cls
            if any(k in name_l for k in ("floor", "ground", "grass", "path")):
                return FloorEntity
            return None

        identified_items: Dict[Tuple[int, int], str] = {}
        agent_pos: Optional[Tuple[int, int]] = None
        agent_entity = None

        # iterate tiles
        for y in range(grid_h):
            for x in range(grid_w):
                tile_img = img[y * tile_h : (y + 1) * tile_h, x * tile_w : (x + 1) * tile_w]

                # classification priority:
                # 1) prototype classifier (if present)
                # 2) template matcher
                # 3) heuristic fallback
                appearance = None
                tile_bytes = tile_img.tobytes()
                if tile_bytes in self._tile_cache:
                    appearance = self._tile_cache[tile_bytes]
                else:
                    if self._proto is not None:
                        appearance = self._proto.predict(tile_img)
                    if appearance is None and templates is not None:
                        appearance = self._classify_tile(tile_img, templates)
                    if appearance is None:
                        appearance = self._heuristic_tile(tile_img)

                    self._tile_cache[tile_bytes] = appearance

                if appearance is None:
                    continue

                ent_cls = to_entity_class(appearance)
                if ent_cls is None or ent_cls is FloorEntity:
                    continue

                game_pos = (x, grid_h - 1 - y)
                ent = ent_cls()
                state.add(game_pos, ent)

                if ent_cls is AgentEntity:
                    agent_pos = game_pos
                    agent_entity = ent
                else:
                    item_type = self._item_type_for_entity(ent_cls)
                    if item_type:
                        identified_items[game_pos] = item_type

        # if agent not found, try info
        if agent_pos is None and isinstance(info, dict):
            maybe_pos = info.get("agent_pos") or info.get("player_pos") or info.get("position")
            if isinstance(maybe_pos, (list, tuple)) and len(maybe_pos) == 2:
                ax, ay = int(maybe_pos[0]), int(maybe_pos[1])
                if 0 <= ax < grid_w and 0 <= ay < grid_h:
                    agent_pos = (ax, ay)
                    agent_entity = AgentEntity()
                    state.add(agent_pos, agent_entity)

        # memory of items under agent
        if agent_pos is not None:
            for pos, item_type in list(self._last_items.items()):
                if pos == agent_pos:
                    if not self._tile_has_item(state, pos, item_type, FloorEntity):
                        self._add_item_by_type(state, pos, item_type, entity_map)
                else:
                    if not self._tile_has_item(state, pos, item_type, FloorEntity):
                        self._last_items.pop(pos, None)

        for pos, item_type in identified_items.items():
            self._last_items[pos] = item_type

        if agent_entity is not None and isinstance(info, dict):
            self._apply_agent_info(agent_entity, info, entity_map)

        return state

    # ---------------- helpers ----------------

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

    def _infer_grid_size(self, info, width_px: int, height_px: int) -> Tuple[Optional[int], Optional[int]]:
        if isinstance(info, dict):
            for w_key, h_key in (("width", "height"), ("grid_width", "grid_height")):
                if w_key in info and h_key in info:
                    try:
                        return int(info[w_key]), int(info[h_key])
                    except Exception:
                        pass
            shape = info.get("grid_shape") or info.get("shape") or info.get("size")
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                try:
                    return int(shape[0]), int(shape[1])
                except Exception:
                    pass

        if width_px > 0 and height_px > 0:
            try:
                import math
                g = math.gcd(width_px, height_px)
                divisors = []
                for d in range(1, int(g ** 0.5) + 1):
                    if g % d == 0:
                        divisors.append(d)
                        if d != g // d:
                            divisors.append(g // d)
                for tile in sorted(divisors, reverse=True):
                    if tile < 8 or tile > 128:
                        continue
                    if width_px % tile == 0 and height_px % tile == 0:
                        gw = width_px // tile
                        gh = height_px // tile
                        if 1 <= gw <= 64 and 1 <= gh <= 64:
                            return gw, gh
            except Exception:
                pass
        return None, None

    def _build_empty_gridstate(self, grid_w: int, grid_h: int):
        try:
            grid_w = max(1, int(grid_w))
            grid_h = max(1, int(grid_h))
            from grid_adventure.grid import GridState
            from grid_adventure.entities import FloorEntity, AgentEntity
            try:
                from grid_adventure.movements import MOVEMENTS
                from grid_adventure.objectives import OBJECTIVES
                movement = MOVEMENTS.get("cardinal") or MOVEMENTS.get("default")
                objective = OBJECTIVES.get("collect_gems_and_exit") or OBJECTIVES.get("default")
                state = GridState(width=grid_w, height=grid_h, movement=movement, objective=objective)
            except Exception:
                state = GridState(width=grid_w, height=grid_h)
            for x in range(grid_w):
                for y in range(grid_h):
                    state.add((x, y), FloorEntity())
            state.add((0, 0), AgentEntity())
            return state
        except Exception:
            return None

    # -------- template pipeline (kept as fallback) --------

    def _load_assets(self) -> Dict[str, List["np.ndarray"]]:
        if self._assets_loaded:
            return self._assets
        assets: Dict[str, List["np.ndarray"]] = {}

        try:
            from grid_adventure.rendering import IMAGE_MAP, DEFAULT_ASSET_ROOT
            asset_root = DEFAULT_ASSET_ROOT
            for name, value in IMAGE_MAP.items():
                for rel_path in self._normalize_image_map_value(value):
                    path = os.path.join(asset_root, rel_path)
                    img = self._read_image(path)
                    if img is not None:
                        assets.setdefault(str(name), []).append(img)
        except Exception:
            pass

        if not assets:
            assets.update(self._load_assets_from_data_dir())

        self._assets = assets
        self._assets_loaded = True
        return assets

    def _load_assets_from_data_dir(self) -> Dict[str, List["np.ndarray"]]:
        assets: Dict[str, List["np.ndarray"]] = {}
        # keep your original path, plus broaden to data/ (so local debug can reuse training set)
        roots = [
            os.path.join(os.getcwd(), "data", "assets"),
            os.path.join(os.getcwd(), "data"),
        ]
        for root in roots:
            if not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fname in filenames:
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        continue
                    path = os.path.join(dirpath, fname)
                    img = self._read_image(path)
                    if img is None:
                        continue
                    stem = os.path.splitext(fname)[0].lower()
                    parent = os.path.basename(os.path.dirname(path)).lower()
                    key = parent if parent not in ("assets", "data") else stem
                    assets.setdefault(key, []).append(img)
        return assets

    def _normalize_image_map_value(self, value) -> List[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        if isinstance(value, dict):
            if "file" in value:
                return [str(value["file"])]
            if "files" in value:
                return [str(v) for v in value["files"]]
        return []

    def _read_image(self, path: str):
        if not os.path.isfile(path):
            return None
        try:
            from PIL import Image
            with Image.open(path) as im:
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

    def _get_templates(self, tile_h: int, tile_w: int) -> dict:
        key = (tile_h, tile_w)
        if key in self._template_cache:
            return self._template_cache[key]

        assets = self._load_assets()
        full_templates: List[Tuple[str, "np.ndarray"]] = []
        overlay_templates: List[Tuple[str, "np.ndarray", "np.ndarray"]] = []
        floor_templates: List["np.ndarray"] = []

        for name, images in assets.items():
            name_l = str(name).lower()
            for img in images:
                resized = self._resize_image(img, tile_w, tile_h)
                if resized.shape[2] == 4:
                    alpha = resized[..., 3:4].astype(np.float32) / 255.0
                else:
                    alpha = None
                rgb = resized[..., :3].astype(np.float32)

                if any(k in name_l for k in ("floor", "ground", "grass", "path")):
                    floor_templates.append(rgb)

                coverage = float(np.mean(alpha > 0)) if alpha is not None else 1.0
                if coverage > 0.95:
                    full_templates.append((name, rgb))
                else:
                    if alpha is None:
                        alpha = np.ones((*rgb.shape[:2], 1), dtype=np.float32)
                    overlay_templates.append((name, rgb, alpha))

        composites: List[Tuple[Optional[str], "np.ndarray"]] = []
        for floor in floor_templates:
            composites.append((None, floor))
            for name, rgb, alpha in overlay_templates:
                comp = rgb * alpha + floor * (1.0 - alpha)
                composites.append((name, comp))

        templates = {"full": full_templates, "composites": composites, "floors": floor_templates}
        self._template_cache[key] = templates
        return templates

    def _classify_tile(self, tile_img: "np.ndarray", templates: dict) -> Optional[str]:
        if templates is None:
            return None

        tile_rgb = tile_img[..., :3].astype(np.float32)
        best_name = None
        best_score = float("inf")

        for name, tmpl_rgb in templates.get("full", []):
            score = float(np.mean((tile_rgb - tmpl_rgb) ** 2)) / (255.0 ** 2)
            if score < best_score:
                best_score = score
                best_name = name

        for name, comp_rgb in templates.get("composites", []):
            score = float(np.mean((tile_rgb - comp_rgb) ** 2)) / (255.0 ** 2)
            if score < best_score:
                best_score = score
                best_name = name

        if best_score > 0.08:
            best_name = None
        return best_name

    def _heuristic_tile(self, tile_img: "np.ndarray") -> Optional[str]:
        rgb = tile_img[..., :3]
        avg_color = rgb.mean(axis=(0, 1))
        r, g, b = avg_color
        if r < 40 and g < 40 and b < 40:
            return "wall"
        if r > 120 and r > 1.3 * g and r > 1.3 * b:
            return "lava"
        if g > max(r, b) and g > 100:
            return "exit"
        if r > 200 and g > 200 and b < 120:
            return "coin"
        if r > 180 and g < 100 and b < 100:
            return "gem"
        if b > 180 and r < 100 and g < 100:
            return "gem"
        if r > 200 and g > 200 and b > 200:
            return "ghost"
        return None

    # -------- item memory helpers --------

    def _item_type_for_entity(self, ent_cls: type) -> Optional[str]:
        name = ent_cls.__name__.lower()
        if "gem" in name:
            return "gem"
        if "coin" in name:
            return "coin"
        if "key" in name:
            return "key"
        if "speed" in name or "boot" in name:
            return "boots"
        if "shield" in name:
            return "shield"
        if "phasing" in name or "ghost" in name:
            return "ghost"
        return None

    def _tile_has_item(self, state, pos: Tuple[int, int], item_type: str, floor_cls) -> bool:
        # Fix: boots/ghost types don't appear literally in entity class names
        for obj in state.objects_at(pos):
            if isinstance(obj, floor_cls):
                continue
            obj_name = type(obj).__name__.lower()

            if item_type == "boots":
                if ("speed" in obj_name) or ("boot" in obj_name):
                    return True
            elif item_type == "ghost":
                if ("phasing" in obj_name) or ("ghost" in obj_name):
                    return True
            else:
                if item_type in obj_name:
                    return True
        return False

    def _add_item_by_type(self, state, pos: Tuple[int, int], item_type: str, entity_map: dict) -> None:
        type_to_key = {
            "gem": "gem",
            "coin": "coin",
            "key": "key",
            "boots": "boots",
            "shield": "shield",
            "ghost": "ghost",
        }
        key = type_to_key.get(item_type)
        if key is None:
            return
        ent_cls = entity_map.get(key)
        if ent_cls is None:
            return
        state.add(pos, ent_cls())

    def _apply_agent_info(self, agent_entity, info: dict, entity_map: dict) -> None:
        keys_count = info.get("keys", info.get("key_count"))
        inv_list = None
        status_list = None

        for attr in ("inventory_list", "inventory"):
            if hasattr(agent_entity, attr):
                inv_list = list(getattr(agent_entity, attr) or [])
                setattr(agent_entity, attr, inv_list)
                break

        for attr in ("status_list", "status"):
            if hasattr(agent_entity, attr):
                status_list = list(getattr(agent_entity, attr) or [])
                setattr(agent_entity, attr, status_list)
                break

        if keys_count is not None and inv_list is not None:
            try:
                keys_count = int(keys_count)
                for _ in range(keys_count):
                    inv_list.append(entity_map["key"]())
            except Exception:
                pass

        speed_turns = info.get("speed", info.get("boots", 0))
        shield_uses = info.get("shield", 0)
        phasing_turns = info.get("ghost", info.get("phasing", 0))
        if status_list is None:
            status_list = inv_list

        if status_list is not None:
            if speed_turns:
                status_list.append(entity_map["boots"]())
            if shield_uses:
                status_list.append(entity_map["shield"]())
            if phasing_turns:
                status_list.append(entity_map["ghost"]())
