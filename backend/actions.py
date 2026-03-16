# ============================================================
# ACTIONS.PY
# All simulation constants:
# — Attack definitions
# — Defense definitions
# — Effectiveness matrix (which defense counters which attack)
# — Reward values
# — Damage values
#
# No logic here. Pure data.
# Imported by: qlearning.py, engine.py, models.py
# ============================================================

from typing import Final
from dataclasses import dataclass


# ----------------------------------------------------------
# ATTACK DEFINITION
# ----------------------------------------------------------

@dataclass(frozen=True)
class Attack:
    id:    int
    name:  str
    short: str
    code:  str


# ----------------------------------------------------------
# DEFENSE DEFINITION
# ----------------------------------------------------------

@dataclass(frozen=True)
class Defense:
    id:    int
    name:  str
    short: str
    code:  str


# ----------------------------------------------------------
# ATTACK REGISTRY
# Order matters — index must match id.
# ----------------------------------------------------------

ATTACKS: Final[list[Attack]] = [
    Attack(id=0, name="SQL Injection", short="SQL-INJ",  code="ATK-01"),
    Attack(id=1, name="DDoS",          short="DDOS",     code="ATK-02"),
    Attack(id=2, name="Phishing",      short="PHISH",    code="ATK-03"),
    Attack(id=3, name="Zero-Day",      short="0-DAY",    code="ATK-04"),
    Attack(id=4, name="Port Scan",     short="PORT-SCN", code="ATK-05"),
    Attack(id=5, name="Ransomware",    short="RANSOM",   code="ATK-06"),
    Attack(id=6, name="MitM",          short="MITM",     code="ATK-07"),
]


# ----------------------------------------------------------
# DEFENSE REGISTRY
# Order matters — index must match id.
# ----------------------------------------------------------

DEFENSES: Final[list[Defense]] = [
    Defense(id=0, name="WAF Rule",       short="WAF",       code="DEF-01"),
    Defense(id=1, name="Rate Limiting",  short="RATE-LIM",  code="DEF-02"),
    Defense(id=2, name="Email Filter",   short="EMAIL-FIL", code="DEF-03"),
    Defense(id=3, name="Patch Deploy",   short="PATCH",     code="DEF-04"),
    Defense(id=4, name="Port Block",     short="PORT-BLK",  code="DEF-05"),
    Defense(id=5, name="Backup/Restore", short="BACKUP",    code="DEF-06"),
    Defense(id=6, name="SSL/TLS",        short="SSL-TLS",   code="DEF-07"),
]


# ----------------------------------------------------------
# DERIVED CONSTANTS
# ----------------------------------------------------------

# Total number of actions (attacks = defenses = N)
N_ACTIONS: Final[int] = len(ATTACKS)

# Flat lists for indexed access
ATTACK_NAMES:   Final[list[str]] = [a.name  for a in ATTACKS]
ATTACK_SHORTS:  Final[list[str]] = [a.short for a in ATTACKS]
ATTACK_CODES:   Final[list[str]] = [a.code  for a in ATTACKS]

DEFENSE_NAMES:  Final[list[str]] = [d.name  for d in DEFENSES]
DEFENSE_SHORTS: Final[list[str]] = [d.short for d in DEFENSES]
DEFENSE_CODES:  Final[list[str]] = [d.code  for d in DEFENSES]


# ----------------------------------------------------------
# EFFECTIVENESS MATRIX
#
# EFFECTIVENESS[attack_id][defense_id] = 1 → defense BLOCKS
# EFFECTIVENESS[attack_id][defense_id] = 0 → attack BREACHES
#
# Each attack has exactly one counter-defense.
# Agents must learn these matchups through Q-learning.
#
# Matrix layout (rows=attacks, cols=defenses):
#
#                WAF  RATE  MAIL  PATC  PORT  BACK  SSL
# ATK-01 SQL      1    0     0     0     0     0     0
# ATK-02 DDoS     0    1     0     0     0     0     0
# ATK-03 Phish    0    0     1     0     0     0     0
# ATK-04 0-Day    0    0     0     1     0     0     0
# ATK-05 PortSc   0    0     0     0     1     0     0
# ATK-06 Ransom   0    0     0     0     0     1     0
# ATK-07 MitM     0    0     0     0     0     0     1
# ----------------------------------------------------------

EFFECTIVENESS: Final[list[list[int]]] = [
    [1, 0, 0, 0, 0, 0, 0],  # SQL Injection  → blocked only by WAF
    [0, 1, 0, 0, 0, 0, 0],  # DDoS           → blocked only by Rate Limiting
    [0, 0, 1, 0, 0, 0, 0],  # Phishing       → blocked only by Email Filter
    [0, 0, 0, 1, 0, 0, 0],  # Zero-Day       → blocked only by Patch Deploy
    [0, 0, 0, 0, 1, 0, 0],  # Port Scan      → blocked only by Port Block
    [0, 0, 0, 0, 0, 1, 0],  # Ransomware     → blocked only by Backup/Restore
    [0, 0, 0, 0, 0, 0, 1],  # MitM           → blocked only by SSL/TLS
]


# ----------------------------------------------------------
# OUTCOME RESOLUTION
# ----------------------------------------------------------

def is_blocked(attack_id: int, defense_id: int) -> bool:
    """
    Return True if the given defense successfully blocks the attack.

    Args:
        attack_id:  Index into ATTACKS (0 to N_ACTIONS-1)
        defense_id: Index into DEFENSES (0 to N_ACTIONS-1)

    Returns:
        True if EFFECTIVENESS[attack_id][defense_id] == 1
    """
    return EFFECTIVENESS[attack_id][defense_id] == 1


# ----------------------------------------------------------
# REWARD VALUES
# ----------------------------------------------------------

# Attacker rewards
REWARD_BREACH:  Final[int] =  1   # Attack gets through
REWARD_BLOCKED: Final[int] = -1   # Attack is countered

# Defender rewards
REWARD_DEFEND_SUCCESS: Final[int] =  1   # Defense counters attack
REWARD_DEFEND_FAIL:    Final[int] = -1   # Defense fails to block


# ----------------------------------------------------------
# DAMAGE VALUES
# HP lost by the target system on a successful breach.
# ----------------------------------------------------------

DAMAGE_MIN: Final[int] = 3
DAMAGE_MAX: Final[int] = 10


# ----------------------------------------------------------
# SYSTEM HEALTH
# ----------------------------------------------------------

HEALTH_START: Final[int] = 100


# ----------------------------------------------------------
# VALIDATION
# Sanity-check the matrix shape at import time.
# Crashes loudly if someone edits ATTACKS/DEFENSES
# without updating EFFECTIVENESS.
# ----------------------------------------------------------

assert len(EFFECTIVENESS) == N_ACTIONS, (
    f"EFFECTIVENESS has {len(EFFECTIVENESS)} rows "
    f"but N_ACTIONS={N_ACTIONS}"
)

for _row_idx, _row in enumerate(EFFECTIVENESS):
    assert len(_row) == N_ACTIONS, (
        f"EFFECTIVENESS row {_row_idx} has {len(_row)} cols "
        f"but N_ACTIONS={N_ACTIONS}"
    )