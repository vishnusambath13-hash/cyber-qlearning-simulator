# ============================================================
# MODELS.PY
# Pydantic v2 request and response schemas.
# Every API endpoint in main.py uses these models to:
#   — Validate incoming request bodies
#   — Serialize outgoing JSON responses
#   — Auto-generate OpenAPI docs at /docs
#
# Depends on: actions.py (for N_ACTIONS constant)
# Imported by: main.py, engine.py
# ============================================================

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator

from backend.actions import (
    N_ACTIONS,
    ATTACK_NAMES,
    ATTACK_SHORTS,
    ATTACK_CODES,
    DEFENSE_NAMES,
    DEFENSE_SHORTS,
    DEFENSE_CODES,
)


# ----------------------------------------------------------
# SHARED SUB-MODELS
# Reusable nested models composed into larger responses.
# ----------------------------------------------------------

class ActionInfo(BaseModel):
    """
    Full description of a single attack or defense action.
    Embedded in agent state responses.
    """
    id:    int   = Field(..., ge=0, lt=N_ACTIONS)
    name:  str
    short: str
    code:  str


class AgentStats(BaseModel):
    """
    Running statistics for one agent (attacker or defender).
    """
    total_actions: int   = Field(default=0, ge=0)
    successes:     int   = Field(default=0, ge=0)
    failures:      int   = Field(default=0, ge=0)
    success_rate:  float = Field(default=0.0, ge=0.0, le=100.0)

    # Per-action frequency counts — length must equal N_ACTIONS
    action_counts: list[int] = Field(
        default_factory=lambda: [0] * N_ACTIONS
    )

    @field_validator("action_counts")
    @classmethod
    def validate_action_counts_length(cls, v: list[int]) -> list[int]:
        if len(v) != N_ACTIONS:
            raise ValueError(
                f"action_counts must have {N_ACTIONS} elements, got {len(v)}"
            )
        return v


class QTableState(BaseModel):
    """
    Serialized Q-table for one agent.
    Shape: N_ACTIONS rows × N_ACTIONS columns.
    Also includes softmax probabilities for the current state.
    """
    # Flat 2D list: qtable[state][action] = Q-value
    qtable: list[list[float]] = Field(...)

    # Softmax probabilities for current state — used by prob bars
    probabilities: list[float] = Field(...)

    @field_validator("qtable")
    @classmethod
    def validate_qtable_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != N_ACTIONS:
            raise ValueError(
                f"qtable must have {N_ACTIONS} rows, got {len(v)}"
            )
        for i, row in enumerate(v):
            if len(row) != N_ACTIONS:
                raise ValueError(
                    f"qtable row {i} must have {N_ACTIONS} cols, got {len(row)}"
                )
        return v

    @field_validator("probabilities")
    @classmethod
    def validate_probabilities_length(cls, v: list[float]) -> list[float]:
        if len(v) != N_ACTIONS:
            raise ValueError(
                f"probabilities must have {N_ACTIONS} elements, got {len(v)}"
            )
        return v


# ----------------------------------------------------------
# REQUEST MODELS
# Bodies sent from frontend to backend.
# ----------------------------------------------------------

class ResetRequest(BaseModel):
    """
    POST /simulation/reset
    No required fields — reset always starts fresh.
    Optional: confirm flag to prevent accidental resets.
    """
    confirm: bool = Field(
        default=True,
        description="Safety flag — must be True to execute reset"
    )


from typing import Optional

class StepRequest(BaseModel):
    """
    POST /simulation/step
    Optional epsilon allows frontend slider to control exploration.
    """

    steps: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Number of simulation steps to run (1–50)"
    )

    epsilon: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override epsilon exploration rate (0–1)"
    )


# ----------------------------------------------------------
# ROUND RESULT MODEL
# Returned for every completed simulation step.
# This is the primary data contract between backend and frontend.
# ----------------------------------------------------------

class RoundResult(BaseModel):
    """
    Full result of one simulation round.
    Returned by POST /simulation/step.
    Contains everything the frontend needs to update the HUD.
    """

    # ---- Round metadata ----
    round:          int   = Field(..., ge=1)
    session_id:     str   = Field(...)    # UUID for this simulation run

    # ---- Attacker action ----
    atk_action_id:   int  = Field(..., ge=0, lt=N_ACTIONS)
    atk_action_name: str
    atk_action_short:str
    atk_action_code: str

    # ---- Defender action ----
    def_action_id:   int  = Field(..., ge=0, lt=N_ACTIONS)
    def_action_name: str
    def_action_short:str
    def_action_code: str

    # ---- Outcome ----
    blocked:     bool
    damage:      int   = Field(..., ge=0)   # 0 if blocked
    atk_reward:  int
    def_reward:  int

    # ---- System state ----
    system_health:  int   = Field(..., ge=0, le=100)
    is_game_over:   bool  = Field(default=False)

    # ---- Running totals ----
    total_rounds:   int   = Field(..., ge=0)
    total_breaches: int   = Field(..., ge=0)
    total_blocks:   int   = Field(..., ge=0)
    breach_rate:    float = Field(..., ge=0.0, le=100.0)
    block_rate:     float = Field(..., ge=0.0, le=100.0)

    # ---- Q-learning state ----
    epsilon:        float = Field(..., ge=0.0, le=1.0)
    atk_qtable:     QTableState
    def_qtable:     QTableState

    # ---- Frequency counts (for bar charts) ----
    atk_counts:     list[int]   = Field(...)
    def_counts:     list[int]   = Field(...)

    # ---- Trend data (breach rate per round, last N rounds) ----
    trend_data:     list[float] = Field(...)

    # ---- Q-learning metadata ----
    current_state:  int   = Field(..., ge=0, lt=N_ACTIONS)
    prev_state:     int   = Field(..., ge=0, lt=N_ACTIONS)


# ----------------------------------------------------------
# SIMULATION STATE MODEL
# Returned by GET /simulation/state
# Snapshot of everything needed to rebuild the HUD on page reload.
# ----------------------------------------------------------

class SimulationState(BaseModel):
    """
    Full simulation state snapshot.
    Returned by GET /simulation/state.
    """

    # ---- Session ----
    session_id:     str
    is_game_over:   bool

    # ---- Round counters ----
    total_rounds:   int   = Field(..., ge=0)
    total_breaches: int   = Field(..., ge=0)
    total_blocks:   int   = Field(..., ge=0)
    breach_rate:    float = Field(..., ge=0.0, le=100.0)
    block_rate:     float = Field(..., ge=0.0, le=100.0)

    # ---- System health ----
    system_health:  int   = Field(..., ge=0, le=100)

    # ---- Q-learning ----
    epsilon:        float = Field(..., ge=0.0, le=1.0)
    current_state:  int   = Field(..., ge=0, lt=N_ACTIONS)
    atk_qtable:     QTableState
    def_qtable:     QTableState

    # ---- Frequency counts ----
    atk_counts:     list[int]
    def_counts:     list[int]

    # ---- Trend ----
    trend_data:     list[float]

    # ---- Last round (None if no rounds run yet) ----
    last_result:    Optional[RoundResult] = Field(default=None)


# ----------------------------------------------------------
# HISTORY ENTRY MODEL
# One row from the database, returned by GET /simulation/history
# ----------------------------------------------------------

class HistoryEntry(BaseModel):
    """
    One round record from the database.
    Lightweight — no Q-tables, just the round facts.
    """
    round:           int
    session_id:      str
    atk_action_id:   int
    atk_action_name: str
    def_action_id:   int
    def_action_name: str
    blocked:         bool
    damage:          int
    system_health:   int
    atk_reward:      int
    def_reward:      int
    epsilon:         float
    timestamp:       str     # ISO 8601 string from SQLite


class HistoryResponse(BaseModel):
    """
    Paginated history response.
    Returned by GET /simulation/history
    """
    session_id:   str
    total_rounds: int
    entries:      list[HistoryEntry]


# ----------------------------------------------------------
# ACTION LABELS MODEL
# Returned by GET /simulation/actions
# Lets the frontend fetch label metadata without hardcoding.
# ----------------------------------------------------------

class ActionsMetadata(BaseModel):
    """
    All attack and defense labels.
    Returned by GET /simulation/actions.
    Frontend fetches this once on load.
    """
    n_actions:      int
    attacks:        list[ActionInfo]
    defenses:       list[ActionInfo]


# ----------------------------------------------------------
# RESET RESPONSE MODEL
# ----------------------------------------------------------

class ResetResponse(BaseModel):
    """
    Confirmation returned by POST /simulation/reset.
    """
    success:    bool
    session_id: str
    message:    str


# ----------------------------------------------------------
# ERROR RESPONSE MODEL
# ----------------------------------------------------------

class ErrorResponse(BaseModel):
    """
    Standardized error envelope.
    Returned on 4xx / 5xx responses.
    """
    error:   str
    detail:  Optional[str] = None
    code:    int


# ----------------------------------------------------------
# HELPER: Build ActionInfo list from actions.py registries
# Used by GET /simulation/actions endpoint in main.py
# ----------------------------------------------------------

def build_actions_metadata() -> ActionsMetadata:
    """
    Construct the ActionsMetadata response from the
    ATTACKS and DEFENSES registries in actions.py.
    """
    attacks = [
        ActionInfo(id=i, name=ATTACK_NAMES[i],
                   short=ATTACK_SHORTS[i], code=ATTACK_CODES[i])
        for i in range(N_ACTIONS)
    ]
    defenses = [
        ActionInfo(id=i, name=DEFENSE_NAMES[i],
                   short=DEFENSE_SHORTS[i], code=DEFENSE_CODES[i])
        for i in range(N_ACTIONS)
    ]
    return ActionsMetadata(
        n_actions=N_ACTIONS,
        attacks=attacks,
        defenses=defenses,
    )