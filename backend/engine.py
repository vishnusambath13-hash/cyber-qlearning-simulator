# ============================================================
# ENGINE.PY
# Simulation orchestrator. The single source of truth for
# all live game state during a server session.
#
# Responsibilities:
#   — Own the two QLearningAgent instances
#   — Own all game state (health, rounds, counts, trend)
#   — Coordinate one full simulation step
#   — Persist results to the database
#   — Build the RoundResult / SimulationState API responses
#   — Handle reset and session management
#
# Depends on: qlearning.py, actions.py, database.py, models.py
# Consumed by: main.py (one global instance: `sim_engine`)
#
# Thread safety: FastAPI runs on a single async event loop.
# All state mutations happen inside awaited coroutines so
# there are no concurrent write races.
# ============================================================

from __future__ import annotations

import random
import string
import uuid
from datetime import datetime, timezone
from typing import Optional

from backend.actions import (
    N_ACTIONS,
    ATTACKS,
    DEFENSES,
    HEALTH_START,
    DAMAGE_MIN,
    DAMAGE_MAX,
)
from backend.database import (
    save_round,
    save_session,
    load_session,
    clear_session_rounds,
    get_latest_session_id,
)
from backend.models import (
    QTableState,
    RoundResult,
    SimulationState,
)
from backend.qlearning import (
    QLearningAgent,
    run_paired_step,
    EPSILON_START,
)


# ----------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------

# Maximum trend data points kept in memory
TREND_MAX_POINTS: int = 40

# Maximum history entries kept in memory for fast access
# (DB is the authoritative store; this is a hot cache)
HISTORY_CACHE_MAX: int = 200


# ----------------------------------------------------------
# SESSION ID GENERATOR
# ----------------------------------------------------------

def _new_session_id() -> str:
    """
    Generate a short, readable session identifier.
    Format: SIM-XXXXXXXX (8 uppercase hex chars)
    Example: SIM-3FA2C81B

    Returns:
        Session ID string
    """
    return "SIM-" + uuid.uuid4().hex[:8].upper()


# ----------------------------------------------------------
# DAMAGE ROLL
# ----------------------------------------------------------

def _roll_damage() -> int:
    """
    Roll random HP damage for a successful breach.

    Returns:
        Integer between DAMAGE_MIN and DAMAGE_MAX inclusive
    """
    return random.randint(DAMAGE_MIN, DAMAGE_MAX)


# ----------------------------------------------------------
# SIMULATION ENGINE
# ----------------------------------------------------------

class SimulationEngine:
    """
    Central orchestrator for the Q-learning cyber battle simulation.

    One instance of this class is created in main.py and shared
    across all request handlers via FastAPI dependency injection.

    State ownership:
        self._attacker    → QLearningAgent (owns Q-table + epsilon)
        self._defender    → QLearningAgent (owns Q-table + epsilon)
        self._state       → dict of all game counters and metadata
    """

    def __init__(self) -> None:
        # Two Q-learning agents
        self._attacker = QLearningAgent(name="attacker")
        self._defender = QLearningAgent(name="defender")

        # All mutable game state in one place
        self._state: dict = self._blank_state()

        # Cache the session creation timestamp
        self._created_at: Optional[str] = None

    # --------------------------------------------------------
    # BLANK STATE FACTORY
    # --------------------------------------------------------

    def _blank_state(self) -> dict:
        """
        Return a fresh state dictionary for a new simulation.
        Called on __init__ and reset().
        """
        return {
            "session_id":     _new_session_id(),
            "round":          0,
            "system_health":  HEALTH_START,
            "current_state":  random.randint(0, N_ACTIONS - 1),
            "total_breaches": 0,
            "total_blocks":   0,
            "atk_counts":     [0] * N_ACTIONS,
            "def_counts":     [0] * N_ACTIONS,
            "trend_data":     [],
            "is_game_over":   False,
            "last_result":    None,    # Last RoundResult object
        }

    # --------------------------------------------------------
    # STARTUP — called from main.py lifespan
    # --------------------------------------------------------

    async def startup(self) -> None:
        """
        Called once when the FastAPI app starts.
        Attempts to resume the latest session from the database.
        If no session exists, starts fresh.
        """
        self._created_at = datetime.now(timezone.utc).isoformat()

        latest_id = await get_latest_session_id()
        if latest_id:
            await self._resume_session(latest_id)
        # else: _blank_state() is already set from __init__

    async def _resume_session(self, session_id: str) -> None:
        """
        Reload a previous session from the database.
        Restores Q-tables, counts, health, and all counters.

        Args:
            session_id: Session ID to load from DB
        """
        row = await load_session(session_id)
        if row is None:
            return  # Session not found — keep blank state

        # Restore game state
        self._state["session_id"]     = row["session_id"]
        self._state["round"]          = row["total_rounds"]
        self._state["system_health"]  = row["system_health"]
        self._state["is_game_over"]   = row["is_game_over"]
        self._state["total_breaches"] = row["total_breaches"]
        self._state["total_blocks"]   = row["total_blocks"]
        self._state["current_state"]  = row["current_state"]
        self._state["atk_counts"]     = row["atk_counts"]
        self._state["def_counts"]     = row["def_counts"]
        self._state["trend_data"]     = row["trend_data"]
        self._created_at              = row.get("created_at")

        # Restore Q-tables into the agents
        self._attacker.qtable  = row["atk_qtable"]
        self._attacker.epsilon = row["epsilon"]
        self._defender.qtable  = row["def_qtable"]
        self._defender.epsilon = row["epsilon"]

    # --------------------------------------------------------
    # STEP — run one simulation round
    # --------------------------------------------------------

    async def step(self) -> RoundResult:
        """
        Execute one full simulation round:

        1. Guard: return early if game is over
        2. Run paired Q-learning step (both agents act + learn)
        3. Apply damage if breach occurred
        4. Update all counters and trend data
        5. Check for game over condition
        6. Persist round to database
        7. Build and return RoundResult

        Returns:
            RoundResult Pydantic model ready for JSON serialization

        Raises:
            RuntimeError: If called when game is already over
        """
        if self._state["is_game_over"]:
            raise RuntimeError(
                "Simulation is over. Call /simulation/reset to start again."
            )

        s = self._state

        # 1. Run Q-learning step
        step_result = run_paired_step(
            attacker=      self._attacker,
            defender=      self._defender,
            current_state= s["current_state"],
        )

        # 2. Increment round counter
        s["round"] += 1

        # 3. Apply damage
        damage = 0
        if not step_result.blocked:
            damage = _roll_damage()
            s["system_health"] = max(0, s["system_health"] - damage)
            s["total_breaches"] += 1
        else:
            s["total_blocks"] += 1

        # 4. Update frequency counters
        s["atk_counts"][step_result.atk_action] += 1
        s["def_counts"][step_result.def_action] += 1

        # 5. Transition Q-learning state
        s["current_state"] = step_result.next_state

        # 6. Update trend data (breach rate % so far)
        total = s["round"]
        breach_rate = round((s["total_breaches"] / total) * 100, 1)
        s["trend_data"].append(breach_rate)
        if len(s["trend_data"]) > TREND_MAX_POINTS:
            s["trend_data"].pop(0)

        # 7. Check game over
        if s["system_health"] <= 0:
            s["is_game_over"] = True

        # 8. Compute derived stats
        block_rate = round((s["total_blocks"] / total) * 100, 1)

        # 9. Persist round to DB (fire-and-forget style —
        #    we await it but don't block the response on DB errors)
        try:
            await save_round(
                session_id=      s["session_id"],
                round_num=       s["round"],
                atk_action_id=   step_result.atk_action,
                atk_action_name= ATTACKS[step_result.atk_action].name,
                def_action_id=   step_result.def_action,
                def_action_name= DEFENSES[step_result.def_action].name,
                blocked=         step_result.blocked,
                damage=          damage,
                system_health=   s["system_health"],
                atk_reward=      step_result.atk_reward,
                def_reward=      step_result.def_reward,
                epsilon=         step_result.epsilon,
            )

            await save_session(
                session_id=     s["session_id"],
                total_rounds=   s["round"],
                total_breaches= s["total_breaches"],
                total_blocks=   s["total_blocks"],
                system_health=  s["system_health"],
                is_game_over=   s["is_game_over"],
                epsilon=        step_result.epsilon,
                current_state=  s["current_state"],
                atk_qtable=     self._attacker.get_qtable(),
                def_qtable=     self._defender.get_qtable(),
                atk_counts=     s["atk_counts"],
                def_counts=     s["def_counts"],
                trend_data=     s["trend_data"],
                created_at=     self._created_at,
            )
        except Exception as db_err:
            # Log but do not crash the simulation on DB errors
            print(f"[ENGINE] DB write error (round {s['round']}): {db_err}")

        # 10. Build response
        result = self._build_round_result(
            step_result=  step_result,
            damage=       damage,
            breach_rate=  breach_rate,
            block_rate=   block_rate,
        )

        s["last_result"] = result
        return result

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    async def reset(self) -> str:
        """
        Reset the simulation to a clean initial state.
        Generates a new session ID.
        Clears in-memory state and resets both Q-learning agents.
        Clears round history for the old session from the DB.

        Returns:
            New session ID string
        """
        old_session_id = self._state["session_id"]

        # Clear old session rounds from DB
        try:
            await clear_session_rounds(old_session_id)
        except Exception as db_err:
            print(f"[ENGINE] DB clear error: {db_err}")

        # Reset agents
        self._attacker.reset()
        self._defender.reset()

        # Reset state
        self._state    = self._blank_state()
        self._created_at = datetime.now(timezone.utc).isoformat()

        # Persist fresh session to DB
        s = self._state
        try:
            await save_session(
                session_id=     s["session_id"],
                total_rounds=   0,
                total_breaches= 0,
                total_blocks=   0,
                system_health=  HEALTH_START,
                is_game_over=   False,
                epsilon=        EPSILON_START,
                current_state=  s["current_state"],
                atk_qtable=     self._attacker.get_qtable(),
                def_qtable=     self._defender.get_qtable(),
                atk_counts=     s["atk_counts"],
                def_counts=     s["def_counts"],
                trend_data=     [],
                created_at=     self._created_at,
            )
        except Exception as db_err:
            print(f"[ENGINE] DB session init error: {db_err}")

        return s["session_id"]

    # --------------------------------------------------------
    # GET STATE — snapshot for GET /simulation/state
    # --------------------------------------------------------

    def get_state(self) -> SimulationState:
        """
        Build a full SimulationState snapshot from current
        in-memory state. Called by GET /simulation/state.

        Returns:
            SimulationState Pydantic model
        """
        s = self._state

        total = s["round"] if s["round"] > 0 else 1
        breach_rate = round((s["total_breaches"] / total) * 100, 1)
        block_rate  = round((s["total_blocks"]  / total) * 100, 1)

        current_st = s["current_state"]

        return SimulationState(
            session_id=     s["session_id"],
            is_game_over=   s["is_game_over"],
            total_rounds=   s["round"],
            total_breaches= s["total_breaches"],
            total_blocks=   s["total_blocks"],
            breach_rate=    breach_rate,
            block_rate=     block_rate,
            system_health=  s["system_health"],
            epsilon=        self._attacker.get_epsilon(),
            current_state=  current_st,
            atk_qtable=     QTableState(
                **self._attacker.to_dict(current_st)
            ),
            def_qtable=     QTableState(
                **self._defender.to_dict(current_st)
            ),
            atk_counts=     list(s["atk_counts"]),
            def_counts=     list(s["def_counts"]),
            trend_data=     list(s["trend_data"]),
            last_result=    s["last_result"],
        )

    # --------------------------------------------------------
    # PROPERTIES — read-only access for main.py
    # --------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._state["session_id"]

    @property
    def is_game_over(self) -> bool:
        return self._state["is_game_over"]

    @property
    def total_rounds(self) -> int:
        return self._state["round"]

    @property
    def system_health(self) -> int:
        return self._state["system_health"]

    # --------------------------------------------------------
    # PRIVATE: BUILD ROUND RESULT
    # --------------------------------------------------------

    def _build_round_result(
        self,
        step_result,
        damage:      int,
        breach_rate: float,
        block_rate:  float,
    ) -> RoundResult:
        """
        Construct a RoundResult Pydantic model from the current
        engine state and this round's StepResult.

        Args:
            step_result: StepResult NamedTuple from qlearning.py
            damage:      HP lost this round (0 if blocked)
            breach_rate: Running breach percentage
            block_rate:  Running block percentage

        Returns:
            RoundResult ready for JSON serialization
        """
        s          = self._state
        current_st = s["current_state"]

        atk = ATTACKS[step_result.atk_action]
        dfn = DEFENSES[step_result.def_action]

        return RoundResult(
            # Round metadata
            round=           s["round"],
            session_id=      s["session_id"],

            # Attacker action
            atk_action_id=   atk.id,
            atk_action_name= atk.name,
            atk_action_short=atk.short,
            atk_action_code= atk.code,

            # Defender action
            def_action_id=   dfn.id,
            def_action_name= dfn.name,
            def_action_short=dfn.short,
            def_action_code= dfn.code,

            # Outcome
            blocked=         step_result.blocked,
            damage=          damage,
            atk_reward=      step_result.atk_reward,
            def_reward=      step_result.def_reward,

            # System state
            system_health=   s["system_health"],
            is_game_over=    s["is_game_over"],

            # Running totals
            total_rounds=    s["round"],
            total_breaches=  s["total_breaches"],
            total_blocks=    s["total_blocks"],
            breach_rate=     breach_rate,
            block_rate=      block_rate,

            # Q-learning state
            epsilon=         step_result.epsilon,
            atk_qtable=      QTableState(
                **self._attacker.to_dict(current_st)
            ),
            def_qtable=      QTableState(
                **self._defender.to_dict(current_st)
            ),

            # Frequency counts
            atk_counts=      list(s["atk_counts"]),
            def_counts=      list(s["def_counts"]),

            # Trend
            trend_data=      list(s["trend_data"]),

            # State transition
            current_state=   current_st,
            prev_state=      step_result.prev_state,
        )