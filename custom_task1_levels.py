from grid_adventure.entities import (
    BoxEntity,
    CoinEntity,
    ExitEntity,
    FloorEntity,
    GemEntity,
    KeyEntity,
    LavaEntity,
    LockedDoorEntity,
    PhasingPowerUpEntity,
    ShieldPowerUpEntity,
    SpeedPowerUpEntity,
    WallEntity,
    create_agent_entity,
)
from grid_adventure.grid import GridState
from grid_adventure.movements import MOVEMENTS
from grid_adventure.objectives import OBJECTIVES


CUSTOM_TURN_LIMIT = 150


def _floors(gridstate: GridState) -> None:
    for y in range(gridstate.height):
        for x in range(gridstate.width):
            gridstate.add((x, y), FloorEntity())


def _border(gridstate: GridState) -> None:
    for x in range(gridstate.width):
        gridstate.add((x, 0), WallEntity())
        gridstate.add((x, gridstate.height - 1), WallEntity())
    for y in range(gridstate.height):
        gridstate.add((0, y), WallEntity())
        gridstate.add((gridstate.width - 1, y), WallEntity())


def _p(x: int, y: int, width: int, mirror: bool) -> tuple[int, int]:
    if mirror:
        return (width - 1 - x, y)
    return (x, y)


def build_level_boss_gauntlet(seed: int = 214) -> GridState:
    """Local harder boss-style Task 1 level.

    Intended sequence for a general planner:
    1. pick up the key on the left side
    2. open the central locked door
    3. collect the first gem in the upper-right area
    4. collect shield to cross lava
    5. collect ghost behind the lava choke
    6. phase through the lower wall to collect the final gem and exit
    """

    gridstate = GridState(
        width=11,
        height=11,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=CUSTOM_TURN_LIMIT,
    )
    _floors(gridstate)
    _border(gridstate)
    mirror = bool(seed % 2)
    width = 11

    gridstate.add(_p(1, 1, width, mirror), create_agent_entity(health=2))
    gridstate.add(_p(9, 9, width, mirror), ExitEntity())

    for y in range(1, 10):
        if y != 5:
            gridstate.add(_p(5, y, width, mirror), WallEntity())
    gridstate.add(_p(5, 5, width, mirror), LockedDoorEntity())

    for x in range(6, 10):
        gridstate.add(_p(x, 8, width, mirror), WallEntity())

    for pos in (
        (7, 2),
        (8, 2),
        (8, 3),
        (6, 4),
        (8, 6),
        (9, 6),
    ):
        gridstate.add(_p(pos[0], pos[1], width, mirror), WallEntity())

    gridstate.add(_p(2, 8, width, mirror), KeyEntity())
    gridstate.add(_p(7, 4, width, mirror), ShieldPowerUpEntity())
    gridstate.add(_p(7, 7, width, mirror), PhasingPowerUpEntity())
    gridstate.add(_p(2, 2, width, mirror), SpeedPowerUpEntity())

    gridstate.add(_p(9, 2, width, mirror), GemEntity())
    gridstate.add(_p(8, 9, width, mirror), GemEntity())

    gridstate.add(_p(7, 6, width, mirror), LavaEntity())
    gridstate.add(_p(6, 7, width, mirror), LavaEntity())

    for pos in ((8, 1), (9, 3), (3, 8)):
        gridstate.add(_p(pos[0], pos[1], width, mirror), CoinEntity())

    gridstate.add(_p(3, 4, width, mirror), BoxEntity())

    return gridstate


def build_level_boss_branching(seed: int = 215) -> GridState:
    """Local branching boss variant with seed-controlled mirroring.

    Core pressure points:
    1. key is mandatory for the central gate
    2. first gem is reachable after the gate
    3. shield sits on a side branch and is favored before ghost
    4. ghost is needed to phase through the lower barrier to the last gem
    5. scattered coins remain visible near the end but should not outrank exit
    """

    gridstate = GridState(
        width=13,
        height=11,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=CUSTOM_TURN_LIMIT,
    )
    _floors(gridstate)
    _border(gridstate)

    mirror = bool(seed % 2)
    width = 13

    gridstate.add(_p(1, 1, width, mirror), create_agent_entity(health=2))
    gridstate.add(_p(11, 9, width, mirror), ExitEntity())

    for y in range(1, 10):
        if y != 5:
            gridstate.add(_p(6, y, width, mirror), WallEntity())
    gridstate.add(_p(6, 5, width, mirror), LockedDoorEntity())

    for pos in (
        (7, 8),
        (8, 8),
        (9, 8),
        (10, 8),
        (11, 8),
        (7, 3),
        (8, 3),
        (9, 3),
        (10, 6),
        (11, 6),
    ):
        gridstate.add(_p(pos[0], pos[1], width, mirror), WallEntity())

    gridstate.add(_p(2, 8, width, mirror), KeyEntity())
    gridstate.add(_p(8, 4, width, mirror), ShieldPowerUpEntity())
    gridstate.add(_p(10, 7, width, mirror), PhasingPowerUpEntity())
    gridstate.add(_p(3, 2, width, mirror), SpeedPowerUpEntity())

    gridstate.add(_p(10, 2, width, mirror), GemEntity())
    gridstate.add(_p(8, 9, width, mirror), GemEntity())

    for pos in ((8, 6), (9, 6), (8, 7)):
        gridstate.add(_p(pos[0], pos[1], width, mirror), LavaEntity())

    for pos in ((9, 1), (11, 3), (10, 4), (9, 9)):
        gridstate.add(_p(pos[0], pos[1], width, mirror), CoinEntity())

    gridstate.add(_p(4, 4, width, mirror), BoxEntity())

    return gridstate


def build_level_fallback_ping_pong(seed: int = 301) -> GridState:
    """No-target probe for weak fallback behavior.

    There is nothing meaningful to plan toward, so the agent falls back to
    `_nearest_safe_move`. A brittle fallback tends to bounce between two cells.
    """

    gridstate = GridState(
        width=9,
        height=7,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=18,
    )
    _floors(gridstate)
    _border(gridstate)
    gridstate.add((2, 3), create_agent_entity())
    return gridstate


def build_level_unreachable_exit_trap(seed: int = 302) -> GridState:
    """Memory / failed-target probe with an unreachable exit."""

    gridstate = GridState(
        width=11,
        height=7,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=30,
    )
    _floors(gridstate)
    _border(gridstate)

    gridstate.add((2, 3), create_agent_entity())
    gridstate.add((8, 3), ExitEntity())

    for y in range(1, 6):
        if y != 3:
            gridstate.add((5, y), WallEntity())
    gridstate.add((5, 3), LockedDoorEntity())

    return gridstate


def build_level_box_two_stage(seed: int = 303) -> GridState:
    """Two-stage box puzzle that pressures dynamic box-state reasoning."""

    gridstate = GridState(
        width=11,
        height=9,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=80,
    )
    _floors(gridstate)
    _border(gridstate)

    gridstate.add((1, 4), create_agent_entity())
    gridstate.add((9, 4), ExitEntity())

    for y in (1, 2, 6, 7):
        gridstate.add((4, y), WallEntity())
        gridstate.add((7, y), WallEntity())

    for pos in ((5, 2), (5, 6), (6, 2), (6, 6)):
        gridstate.add(pos, WallEntity())

    gridstate.add((4, 4), BoxEntity())
    gridstate.add((7, 4), BoxEntity())

    return gridstate


def build_level_boss_shifted_coin(seed: int = 304) -> GridState:
    """Intro-boss-like map with a slightly shifted signature.

    This keeps the broad boss mechanics but breaks the exact hardcoded lookup.
    """

    gridstate = GridState(
        width=7,
        height=7,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=CUSTOM_TURN_LIMIT,
    )
    _floors(gridstate)

    gridstate.add((0, 0), create_agent_entity(health=1))
    gridstate.add((0, 6), ExitEntity())

    for pos in (
        (3, 0),
        (0, 1),
        (1, 1),
        (3, 1),
        (5, 1),
        (3, 2),
        (5, 2),
        (1, 3),
        (1, 4),
        (3, 4),
        (5, 4),
        (1, 5),
        (2, 5),
        (3, 5),
        (5, 5),
    ):
        gridstate.add(pos, WallEntity())

    gridstate.add((2, 1), BoxEntity())

    gridstate.add((0, 5), GemEntity())
    gridstate.add((6, 3), GemEntity())

    for pos in ((1, 2), (4, 2), (3, 3), (6, 5), (2, 6), (4, 6)):
        gridstate.add(pos, CoinEntity())

    gridstate.add((0, 2), SpeedPowerUpEntity())
    gridstate.add((2, 3), PhasingPowerUpEntity())
    gridstate.add((4, 0), ShieldPowerUpEntity())

    gridstate.add((4, 4), KeyEntity())
    gridstate.add((1, 6), LockedDoorEntity())
    gridstate.add((5, 3), LavaEntity())

    return gridstate


def build_level_optional_coin_pocket(seed: int = 305) -> GridState:
    """Optional-coin probe where a small pocket of coins is worth the detour."""

    gridstate = GridState(
        width=11,
        height=7,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=50,
    )
    _floors(gridstate)
    _border(gridstate)

    gridstate.add((1, 3), create_agent_entity())
    gridstate.add((9, 3), ExitEntity())

    for pos in ((5, 1), (6, 1), (5, 2), (6, 2)):
        gridstate.add(pos, CoinEntity())

    return gridstate


def build_level_lava_commitment_trap(seed: int = 306) -> GridState:
    """Lava-budget probe where shield-first is required to finish safely."""

    gridstate = GridState(
        width=11,
        height=7,
        movement=MOVEMENTS["cardinal"],
        objective=OBJECTIVES["collect_gems_and_exit"],
        seed=seed,
        turn_limit=60,
    )
    _floors(gridstate)
    _border(gridstate)

    gridstate.add((1, 3), create_agent_entity(health=5))
    gridstate.add((1, 1), ExitEntity())
    gridstate.add((2, 1), ShieldPowerUpEntity())
    gridstate.add((9, 3), GemEntity())

    for x in range(3, 10):
        gridstate.add((x, 2), WallEntity())
        gridstate.add((x, 4), WallEntity())

    gridstate.add((5, 3), LavaEntity())
    gridstate.add((6, 3), LavaEntity())

    return gridstate
