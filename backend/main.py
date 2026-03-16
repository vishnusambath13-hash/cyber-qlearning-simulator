# ============================================================
# MAIN.PY
# FastAPI application entry point.
#
# Responsibilities:
#   — App lifecycle (startup / shutdown)
#   — CORS configuration (allow frontend origin)
#   — All API route definitions
#   — Static file serving for the frontend
#   — Global exception handling
#   — Single SimulationEngine instance (dependency injection)
#
# Run with:
#   uvicorn backend.main:app --reload --port 8000
#
# Depends on: engine.py, models.py, database.py, actions.py
# ============================================================

from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.database import init_database_sync, load_history, count_rounds
from backend.engine import SimulationEngine
from backend.models import (
    ActionsMetadata,
    ErrorResponse,
    HistoryResponse,
    ResetRequest,
    ResetResponse,
    RoundResult,
    SimulationState,
    StepRequest,
    build_actions_metadata,
)


# ----------------------------------------------------------
# PATHS
# ----------------------------------------------------------

_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend")


# ----------------------------------------------------------
# GLOBAL ENGINE INSTANCE
# One engine for the lifetime of the server process.
# All route handlers share this instance.
# ----------------------------------------------------------

engine = SimulationEngine()


# ----------------------------------------------------------
# DEPENDENCY
# Inject the engine into route handlers via FastAPI DI.
# This makes routes testable — swap the engine in tests.
# ----------------------------------------------------------

def get_engine() -> SimulationEngine:
    return engine


EngineDep = Annotated[SimulationEngine, Depends(get_engine)]


# ----------------------------------------------------------
# LIFESPAN
# Replaces deprecated @app.on_event("startup").
# Runs init_database_sync and engine.startup() before
# the server accepts any requests.
# ----------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    # 1. Create DB file and tables (sync, before event loop work)
    init_database_sync()

    # 2. Start engine — attempts to resume latest session from DB
    await engine.startup()

    print(f"[STARTUP] Simulation engine ready.")
    print(f"[STARTUP] Session: {engine.session_id}")
    print(f"[STARTUP] Rounds completed: {engine.total_rounds}")
    print(f"[STARTUP] Serving frontend from: {_FRONTEND_DIR}")

    yield  # Server runs here

    # --- SHUTDOWN ---
    print("[SHUTDOWN] Server stopping. State persisted in DB.")


# ----------------------------------------------------------
# APP FACTORY
# ----------------------------------------------------------

app = FastAPI(
    title="Cyber Battle Q-Learning Simulator",
    description=(
        "REST API for a cybersecurity simulation where an Attacker "
        "and Defender agent learn strategies using Q-learning (Q-tables). "
        "No neural networks. No deep learning. Pure tabular RL."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ----------------------------------------------------------
# CORS
# Allow the frontend (served on any port during dev)
# to call the backend on port 8000.
# ----------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # live-server / http.server
        "http://localhost:5500",   # VS Code Live Server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "null",                    # file:// origin (opening HTML directly)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------
# STATIC FILES
# Serve the frontend directory at /app
# index.html is served explicitly at / below.
# ----------------------------------------------------------

if os.path.isdir(_FRONTEND_DIR):
    app.mount(
        "/static",
        StaticFiles(directory=_FRONTEND_DIR),
        name="static",
    )


# ----------------------------------------------------------
# GLOBAL EXCEPTION HANDLER
# Returns a consistent JSON error envelope on unhandled errors.
# ----------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc:     Exception,
) -> JSONResponse:
    tb = traceback.format_exc()
    print(f"[ERROR] Unhandled exception on {request.url}:\n{tb}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            code=500,
        ).model_dump(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Request,
    exc:     HTTPException,
) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=None,
            code=exc.status_code,
        ).model_dump(),
    )


# ----------------------------------------------------------
# ROOT — serve frontend index.html
# ----------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """
    Serve the frontend index.html at the root URL.
    Allows the frontend to be accessed at http://localhost:8000/
    without a separate static server during development.
    """
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(
            status_code=404,
            detail="Frontend not found. Ensure frontend/index.html exists.",
        )
    return FileResponse(index_path)


# ----------------------------------------------------------
# HEALTH CHECK
# ----------------------------------------------------------

@app.get("/health", tags=["Meta"])
async def health_check() -> dict:
    """
    Simple health check endpoint.
    Returns server status and current session info.
    """
    return {
        "status":        "ok",
        "session_id":    engine.session_id,
        "total_rounds":  engine.total_rounds,
        "system_health": engine.system_health,
        "is_game_over":  engine.is_game_over,
    }


# ----------------------------------------------------------
# GET /simulation/actions
# Returns attack and defense label metadata.
# Frontend fetches this once on load.
# ----------------------------------------------------------

@app.get(
    "/simulation/actions",
    response_model=ActionsMetadata,
    tags=["Simulation"],
    summary="Get attack and defense action labels",
)
async def get_actions() -> ActionsMetadata:
    """
    Return all attack and defense definitions.
    The frontend uses this to populate labels without
    hardcoding names in JavaScript.
    """
    return build_actions_metadata()


# ----------------------------------------------------------
# GET /simulation/state
# Returns full current simulation state snapshot.
# ----------------------------------------------------------

@app.get(
    "/simulation/state",
    response_model=SimulationState,
    tags=["Simulation"],
    summary="Get current simulation state",
)
async def get_state(eng: EngineDep) -> SimulationState:
    """
    Return a complete snapshot of the current simulation state.

    Includes:
    - Round counters and health
    - Both Q-tables and softmax probabilities
    - Action frequency counts
    - Trend data
    - Last round result (if any rounds have been run)

    Used by the frontend on page load to rebuild the HUD.
    """
    return eng.get_state()


# ----------------------------------------------------------
# POST /simulation/step
# Run one (or more) simulation rounds.
# ----------------------------------------------------------

@app.post(
    "/simulation/step",
    response_model=RoundResult,
    tags=["Simulation"],
    summary="Run one simulation step",
)
async def run_step(
    body: StepRequest,
    eng:  EngineDep,
) -> RoundResult:
    """
    Execute one simulation round:
    1. Attacker selects action via epsilon-greedy Q-learning
    2. Defender selects action via epsilon-greedy Q-learning
    3. Outcome resolved via effectiveness matrix
    4. Both Q-tables updated via Bellman equation
    5. Round persisted to database
    6. Full RoundResult returned

    If `steps` > 1 in the request body, runs that many rounds
    and returns only the final round's result.

    Raises:
        409: If the simulation is already game over
        500: On unexpected engine errors
    """
    if eng.is_game_over:
        raise HTTPException(
            status_code=409,
            detail="System compromised. Call /simulation/reset to restart.",
        )

        # Apply epsilon from frontend slider if provided
    if body.epsilon is not None:
        eng._attacker.epsilon = body.epsilon
        eng._defender.epsilon = body.epsilon

    last_result: RoundResult | None = None

    try:
        for _ in range(body.steps):
            last_result = await eng.step()
            # Stop early if game ends mid-batch
            if last_result.is_game_over:
                break
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Engine error: {str(e)}",
        )

    return last_result


# ----------------------------------------------------------
# POST /simulation/reset
# Reset simulation to initial state.
# ----------------------------------------------------------

@app.post(
    "/simulation/reset",
    response_model=ResetResponse,
    tags=["Simulation"],
    summary="Reset the simulation",
)
async def reset_simulation(
    body: ResetRequest,
    eng:  EngineDep,
) -> ResetResponse:
    """
    Reset the simulation:
    - Clears both Q-tables (reset to zeros)
    - Resets system health to 100
    - Resets all counters and history
    - Generates a new session ID
    - Clears round history from database
    - Persists fresh session to database

    The `confirm: true` flag in the request body is required
    to prevent accidental resets.

    Raises:
        400: If confirm is not True
    """
    if not body.confirm:
        raise HTTPException(
            status_code=400,
            detail="Reset requires confirm: true in request body.",
        )

    new_session_id = await eng.reset()

    return ResetResponse(
        success=    True,
        session_id= new_session_id,
        message=    (
            f"Simulation reset. "
            f"New session: {new_session_id}. "
            f"Q-tables cleared. Health restored to 100."
        ),
    )


# ----------------------------------------------------------
# GET /simulation/history
# Fetch round history from the database.
# ----------------------------------------------------------

@app.get(
    "/simulation/history",
    response_model=HistoryResponse,
    tags=["Simulation"],
    summary="Get round history",
)
async def get_history(
    eng:    EngineDep,
    limit:  Annotated[int, Query(ge=1,  le=200)] = 50,
    offset: Annotated[int, Query(ge=0)]          = 0,
) -> HistoryResponse:
    """
    Return paginated round history from the database.

    Query parameters:
        limit:  Max records to return (1–200, default 50)
        offset: Pagination offset (default 0)

    Records are returned newest-first (highest round number first).
    """
    session_id = eng.session_id

    try:
        entries = await load_history(
            session_id= session_id,
            limit=      limit,
            offset=     offset,
        )
        total = await count_rounds(session_id)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}",
        )

    return HistoryResponse(
        session_id=   session_id,
        total_rounds= total,
        entries=      entries,
    )


# ----------------------------------------------------------
# GET /simulation/history/{round_num}
# Fetch a single round record by round number.
# ----------------------------------------------------------

@app.get(
    "/simulation/history/{round_num}",
    tags=["Simulation"],
    summary="Get a specific round by number",
)
async def get_round(
    round_num: int,
    eng:       EngineDep,
) -> dict:
    """
    Return a single round record from the database.

    Args:
        round_num: The round number to fetch (1-based)

    Raises:
        404: If the round does not exist in the current session
    """
    session_id = eng.session_id

    try:
        entries = await load_history(
            session_id= session_id,
            limit=      200,
            offset=     0,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}",
        )

    match = next(
        (e for e in entries if e["round"] == round_num),
        None,
    )

    if match is None:
        raise HTTPException(
            status_code=404,
            detail=f"Round {round_num} not found in session {session_id}.",
        )

    return match