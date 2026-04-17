"""
Smoke-test all 4 curriculum stages without starting Ray or training.
Run with:  python test_curriculum_stages.py
Each stage prints PASS or FAIL with a reason.
"""

import sys
import traceback
import numpy as np

# --- helpers ---------------------------------------------------------------

OBS_336 = np.random.randn(336).astype(np.float32)
OBS_672 = np.random.randn(672).astype(np.float32)


def check(name, fn, *args):
    try:
        result = fn(*args)
        arr = np.asarray(result)
        assert arr.shape == (3,), f"expected shape (3,), got {arr.shape}"
        assert arr.dtype in (np.int32, np.int64, np.int_), f"expected int dtype, got {arr.dtype}"
        assert all(0 <= x <= 2 for x in arr), f"action out of range [0,2]: {arr}"
        print(f"  PASS  {name}  →  action={arr}")
        return True
    except Exception as e:
        print(f"  FAIL  {name}  →  {e}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Stage 0: still
# ---------------------------------------------------------------------------
print("\n=== Stage 0: still ===")
from ppo_curriculum import _still_opponent
ok0 = check("still", _still_opponent, OBS_336)


# ---------------------------------------------------------------------------
# Stage 1: random
# ---------------------------------------------------------------------------
print("\n=== Stage 1: random ===")
from ppo_curriculum import _random_opponent
ok1 = check("random", _random_opponent, OBS_336)


# ---------------------------------------------------------------------------
# Stage 2: frozen self  (loads PPO_shaped checkpoint_002500)
# ---------------------------------------------------------------------------
print("\n=== Stage 2: frozen self ===")
import pickle, os
from ppo_curriculum import _make_frozen_opponent, NumpyMLP

SHAPED_CKPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "HU_PPO2_agent/ray_results/PPO_shaped/checkpoint-2500",
)
try:
    with open(SHAPED_CKPT, "rb") as f:
        ckpt = pickle.load(f)
    state_dict = pickle.loads(ckpt["worker"])["state"]["default_policy"]
    state_dict.pop("_optimizer_variables", None)

    # Check expected weight shapes
    w0 = state_dict["_hidden_layers.0._model.0.weight"]
    print(f"  hidden[0] shape: {w0.shape}  (expected (512, 672))")
    w_out = state_dict["_logits._model.0.weight"]
    print(f"  logits shape:    {w_out.shape}  (expected (18, 512))")

    frozen_opp = _make_frozen_opponent(state_dict)
    ok2 = check("frozen-self", frozen_opp, OBS_336)
except Exception as e:
    print(f"  FAIL  frozen-self  →  {e}")
    traceback.print_exc()
    ok2 = False


# ---------------------------------------------------------------------------
# Stage 3: CEIA
# ---------------------------------------------------------------------------
print("\n=== Stage 3: CEIA ===")
from ppo_curriculum import _load_ceia_opponent, CEIA_CHECKPOINT

if not os.path.exists(CEIA_CHECKPOINT):
    print(f"  SKIP  CEIA checkpoint not found at:\n  {CEIA_CHECKPOINT}")
    ok3 = None
else:
    try:
        with open(CEIA_CHECKPOINT, "rb") as f:
            ckpt = pickle.load(f)
        state_dict = pickle.loads(ckpt["worker"])["state"]["default"]
        state_dict.pop("_optimizer_variables", None)

        w0 = state_dict["_hidden_layers.0._model.0.weight"]
        print(f"  hidden[0] shape: {w0.shape}  (expected (256, 336))")
        w_out = state_dict["_logits._model.0.weight"]
        print(f"  logits shape:    {w_out.shape}  (expected (9, 256))")

        ceia_opp = _load_ceia_opponent()
        ok3 = check("CEIA", ceia_opp, OBS_336)
    except Exception as e:
        print(f"  FAIL  CEIA  →  {e}")
        traceback.print_exc()
        ok3 = False


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n=== Summary ===")
results = {
    "Stage 0 (still)":       ok0,
    "Stage 1 (random)":      ok1,
    "Stage 2 (frozen-self)": ok2,
    "Stage 3 (CEIA)":        ok3,
}
all_ok = True
for name, ok in results.items():
    status = "PASS" if ok else ("SKIP" if ok is None else "FAIL")
    print(f"  {status}  {name}")
    if ok is False:
        all_ok = False

print()
if all_ok:
    print("All stages OK — safe to run ppo_curriculum.py")
else:
    print("One or more stages FAILED — fix before training")
    sys.exit(1)
