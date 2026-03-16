# ============================================================
# QLEARNING.PY
# Pure Q-learning implementation. No FastAPI. No database.
# No side effects. Only math and state.
#
# Implements:
#   — Q-table initialization
#   — Epsilon-greedy action selection
#   — Softmax probability distribution
#   — Q-value update rule (Bellman equation)
#   — Epsilon decay
#
# Depends on: actions.py (N_ACTIONS, reward constants)
# Consumed by: engine.py
# ============================================================

from __future__ import annotations

import math
import random
from copy import deepcopy
from typing import NamedTuple

from backend.actions import (
    N_ACTIONS,
    REWARD_BREACH,
    REWARD_BLOCKED,
    REWARD_DEFEND_SUCCESS,
    REWARD_DEFEND_FAIL,
    is_blocked,
)


# ----------------------------------------------------------
# HYPERPARAMETERS
# ----------------------------------------------------------

ALPHA:          float = 0.1    # Learning rate
GAMMA:          float = 0.9    # Discount factor
EPSILON_START:  float = 0.9    # Initial exploration rate
EPSILON_MIN:    float = 0.05   # Floor — never stop exploring
EPSILON_DECAY:  float = 0.995  # Multiplicative decay per step
SOFTMAX_TEMP:   float = 0.5    # Sharpness of softmax distribution


# ----------------------------------------------------------
# TYPE ALIASES
# ----------------------------------------------------------

# A Q-table is a list of N rows, each row has N float values.
# qtable[state][action] = Q-value
QTable = list[list[float]]


# ----------------------------------------------------------
# STEP RESULT
# Returned by QLearningEngine.step().
# A NamedTuple so fields are accessible by name and by index.
# ----------------------------------------------------------

class StepResult(NamedTuple):
    atk_action:  int    # Index of attack chosen (0 to N-1)
    def_action:  int    # Index of defense chosen (0 to N-1)
    blocked:     bool   # True if defense countered the attack
    atk_reward:  int    # Reward assigned to attacker
    def_reward:  int    # Reward assigned to defender
    prev_state:  int    # Q-learning state before this step
    next_state:  int    # Q-learning state after this step
    epsilon:     float  # Epsilon value after decay this step


# ----------------------------------------------------------
# PURE FUNCTIONS
# All stateless. Take inputs, return outputs. No mutations.
# ----------------------------------------------------------

def build_qtable(n: int = N_ACTIONS) -> QTable:
    """
    Create a fresh N×N Q-table filled with zeros.

    Args:
        n: Number of states and actions (default: N_ACTIONS)

    Returns:
        list[list[float]] of shape [n][n], all zeros
    """
    return [[0.0] * n for _ in range(n)]


def argmax(row: list[float]) -> int:
    """
    Return the index of the maximum value in a list.
    Ties broken by lowest index (stable, deterministic).

    Args:
        row: List of Q-values for one state

    Returns:
        Index of the highest Q-value
    """
    best_idx = 0
    best_val = row[0]
    for i in range(1, len(row)):
        if row[i] > best_val:
            best_val = row[i]
            best_idx = i
    return best_idx


def epsilon_greedy(
    qtable:  QTable,
    state:   int,
    epsilon: float,
) -> int:
    """
    Epsilon-greedy action selection.

    With probability epsilon  → choose a random action (explore).
    With probability 1-epsilon → choose argmax Q(state, ·) (exploit).

    Args:
        qtable:  The agent's Q-table
        state:   Current state index
        epsilon: Current exploration rate (0.0 to 1.0)

    Returns:
        Chosen action index (0 to N_ACTIONS-1)
    """
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)   # explore
    return argmax(qtable[state])                   # exploit


def softmax(row: list[float], temperature: float = SOFTMAX_TEMP) -> list[float]:
    """
    Convert a row of Q-values into a probability distribution
    using the softmax function with a temperature parameter.

    Lower temperature → sharper distribution (more confident).
    Higher temperature → flatter distribution (more uniform).

    Numerically stable: subtracts max(row) before exponentiation
    to prevent overflow on large Q-values.

    Args:
        row:         List of Q-values for one state
        temperature: Softmax temperature (default: SOFTMAX_TEMP)

    Returns:
        List of probabilities summing to 1.0
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0, got {temperature}")

    # Numerical stability: shift by max value
    max_val = max(row)
    exps    = [math.exp((v - max_val) / temperature) for v in row]
    total   = sum(exps)

    if total == 0:
        # Fallback: uniform distribution
        return [1.0 / len(row)] * len(row)

    return [e / total for e in exps]


def update_q(
    qtable:     QTable,
    state:      int,
    action:     int,
    reward:     float,
    next_state: int,
    alpha:      float = ALPHA,
    gamma:      float = GAMMA,
) -> None:
    """
    Apply the Q-learning update rule (Bellman equation) in-place.

    Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]

    Where:
        s  = current state
        a  = action taken
        r  = reward received
        s' = next state
        α  = learning rate
        γ  = discount factor

    Args:
        qtable:     Q-table to update (mutated in place)
        state:      Current state s
        action:     Action taken a
        reward:     Reward received r
        next_state: Next state s'
        alpha:      Learning rate (default: ALPHA)
        gamma:      Discount factor (default: GAMMA)
    """
    current_q   = qtable[state][action]
    max_next_q  = max(qtable[next_state])
    td_target   = reward + gamma * max_next_q
    td_error    = td_target - current_q
    qtable[state][action] = current_q + alpha * td_error


def decay_epsilon(epsilon: float) -> float:
    """
    Apply multiplicative epsilon decay, clamped to EPSILON_MIN.

    Args:
        epsilon: Current epsilon value

    Returns:
        New epsilon value after decay
    """
    return max(EPSILON_MIN, epsilon * EPSILON_DECAY)


def get_qtable_snapshot(qtable: QTable) -> list[list[float]]:
    """
    Return a deep copy of a Q-table for safe external read.
    Prevents callers from accidentally mutating the live table.

    Args:
        qtable: Q-table to snapshot

    Returns:
        Deep copy as list[list[float]]
    """
    return deepcopy(qtable)


def get_probabilities(qtable: QTable, state: int) -> list[float]:
    """
    Return softmax probabilities for a given state.
    Used to populate the probability bars in the frontend.

    Args:
        qtable: Agent's Q-table
        state:  Current state index

    Returns:
        List of N probabilities summing to 1.0
    """
    return softmax(qtable[state])


# ----------------------------------------------------------
# Q-LEARNING ENGINE CLASS
# Stateful wrapper around the pure functions above.
# One instance per agent (attacker and defender each get one).
# engine.py creates two of these and coordinates them.
# ----------------------------------------------------------

class QLearningAgent:
    """
    Stateful Q-learning agent.

    Owns:
        — Q-table (N×N float matrix)
        — Current epsilon value

    Does NOT own:
        — Game state (health, round count) → engine.py
        — Database writes                  → database.py
        — HTTP concerns                    → main.py
    """

    def __init__(self, name: str) -> None:
        """
        Initialize agent with a fresh Q-table and full epsilon.

        Args:
            name: Human-readable label ("attacker" or "defender")
        """
        self.name:    str     = name
        self.qtable:  QTable  = build_qtable()
        self.epsilon: float   = EPSILON_START

    # --------------------------------------------------------
    # ACTION SELECTION
    # --------------------------------------------------------

    def select_action(self, state: int) -> int:
        """
        Choose an action for the given state using epsilon-greedy.

        Args:
            state: Current Q-learning state index

        Returns:
            Chosen action index
        """
        return epsilon_greedy(self.qtable, state, self.epsilon)

    # --------------------------------------------------------
    # LEARNING
    # --------------------------------------------------------

    def learn(
        self,
        state:      int,
        action:     int,
        reward:     float,
        next_state: int,
    ) -> None:
        """
        Update Q-table for one (state, action, reward, next_state) tuple.

        Args:
            state:      State when action was taken
            action:     Action that was taken
            reward:     Reward received
            next_state: State transitioned to
        """
        update_q(
            qtable=     self.qtable,
            state=      state,
            action=     action,
            reward=     reward,
            next_state= next_state,
        )

    def decay(self) -> None:
        """
        Decay epsilon by one step.
        Called once per round after both agents have learned.
        """
        self.epsilon = decay_epsilon(self.epsilon)

    # --------------------------------------------------------
    # READ-ONLY ACCESS
    # --------------------------------------------------------

    def get_qtable(self) -> list[list[float]]:
        """
        Return a deep copy of the Q-table (safe for serialization).
        """
        return get_qtable_snapshot(self.qtable)

    def get_probabilities(self, state: int) -> list[float]:
        """
        Return softmax action probabilities for the given state.

        Args:
            state: Current state index

        Returns:
            List of N probabilities summing to 1.0
        """
        return get_probabilities(self.qtable, state)

    def get_epsilon(self) -> float:
        """Return current epsilon value."""
        return self.epsilon

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self) -> None:
        """
        Reset Q-table and epsilon to initial values.
        Called by engine.py on simulation reset.
        """
        self.qtable  = build_qtable()
        self.epsilon = EPSILON_START

    # --------------------------------------------------------
    # SERIALIZATION HELPERS
    # Used by engine.py when building the API response.
    # --------------------------------------------------------

    def to_dict(self, state: int) -> dict:
        """
        Serialize agent Q-learning state for API response.

        Args:
            state: Current Q-learning state (for probabilities)

        Returns:
            dict with 'qtable' and 'probabilities' keys
        """
        return {
            "qtable":        self.get_qtable(),
            "probabilities": self.get_probabilities(state),
        }


# ----------------------------------------------------------
# PAIRED STEP
# A single function that runs one full round for BOTH agents.
# engine.py calls this instead of coordinating agents manually.
# ----------------------------------------------------------

def run_paired_step(
    attacker:      QLearningAgent,
    defender:      QLearningAgent,
    current_state: int,
) -> StepResult:
    """
    Run one complete Q-learning step for both agents:

    1. Attacker selects action via epsilon-greedy
    2. Defender selects action via epsilon-greedy
    3. Resolve outcome (blocked or breach)
    4. Assign rewards to both agents
    5. Determine next state (= attacker's action index)
    6. Update both Q-tables
    7. Decay epsilon for both agents

    Args:
        attacker:      The attacking QLearningAgent
        defender:      The defending QLearningAgent
        current_state: Current shared Q-learning state index

    Returns:
        StepResult NamedTuple with all round data
    """
    prev_state = current_state

    # 1 & 2. Both agents select actions independently
    atk_action = attacker.select_action(prev_state)
    def_action = defender.select_action(prev_state)

    # 3. Resolve outcome using effectiveness matrix
    blocked = is_blocked(atk_action, def_action)

    # 4. Assign rewards
    atk_reward = REWARD_BLOCKED       if blocked else REWARD_BREACH
    def_reward = REWARD_DEFEND_SUCCESS if blocked else REWARD_DEFEND_FAIL

    # 5. Next state = last attacker action
    #    This encodes "what did the attacker just do?" as state,
    #    so both agents can condition their next move on it.
    next_state = atk_action

    # 6. Both agents update their Q-tables
    attacker.learn(prev_state, atk_action, atk_reward, next_state)
    defender.learn(prev_state, def_action, def_reward, next_state)

    # 7. Decay epsilon (shared — both agents explore at same rate)
    attacker.decay()
    defender.decay()

    return StepResult(
        atk_action=  atk_action,
        def_action=  def_action,
        blocked=     blocked,
        atk_reward=  atk_reward,
        def_reward=  def_reward,
        prev_state=  prev_state,
        next_state=  next_state,
        epsilon=     attacker.get_epsilon(),  # same for both after decay
    )