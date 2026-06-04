"""De-risking experiment: does JAX autodiff through MJX FK reproduce MuJoCo ground truth?

This validates the core thesis of the JAX-fabrics library: that the two derivative
terms every fabric needs can come straight from autodiff instead of hand-derivation.

  J      = jax.jacobian(fk)        vs   mujoco.mj_jac{Body,Site}   (first order)
  J_dot q_dot = nested jax.jvp     vs   central finite differences  (the curvature term)

If the diffs are ~1e-9 (x64), MJX-as-kinematics-provider is locked in.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

jax.config.update("jax_enable_x64", True)  # match MuJoCo float64 for a clean comparison

XML = "/Users/huzheyuan/Documents/kinova/mujoco_menagerie/kinova_gen3/gen3.xml"

print("jax devices:", jax.devices())

# ---------- ground-truth MuJoCo model ----------
mj_model = mujoco.MjModel.from_xml_path(XML)
mj_data = mujoco.MjData(mj_model)
print(f"nq={mj_model.nq} nv={mj_model.nv} njnt={mj_model.njnt} "
      f"nbody={mj_model.nbody} nsite={mj_model.nsite}")
jnt_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
             for i in range(mj_model.njnt)]
print("joints:", jnt_names)

# choose an end-effector frame: prefer a site, else the last body
if mj_model.nsite > 0:
    site_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
                  for i in range(mj_model.nsite)]
    print("sites:", site_names)
    ee_kind, ee_id = "site", mj_model.nsite - 1
    ee_name = site_names[ee_id]
else:
    ee_kind, ee_id = "body", mj_model.nbody - 1
    ee_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_id)
print(f"EE frame: {ee_kind} '{ee_name}' (id {ee_id})\n")

# ---------- MJX (JAX) forward kinematics ----------
mjx_model = mjx.put_model(mj_model)

def fk_jax(q):
    d = mjx.make_data(mjx_model).replace(qpos=q)
    d = mjx.kinematics(mjx_model, d)
    return d.site_xpos[ee_id] if ee_kind == "site" else d.xpos[ee_id]

def jdotqd_jax(q, qd):
    f1 = lambda qq: jax.jvp(fk_jax, (qq,), (qd,))[1]   # q -> J(q) qd
    return jax.jvp(f1, (q,), (qd,))[1]                 # directional deriv along qd => J_dot q_dot

fk_j = jax.jit(fk_jax)
J_j = jax.jit(jax.jacobian(fk_jax))
jdq_j = jax.jit(jdotqd_jax)

# ---------- MuJoCo ground truth ----------
def mj_pos_and_J(q):
    mj_data.qpos[:] = q
    mujoco.mj_kinematics(mj_model, mj_data)
    mujoco.mj_comPos(mj_model, mj_data)  # cdof needed by mj_jac
    jacp = np.zeros((3, mj_model.nv)); jacr = np.zeros((3, mj_model.nv))
    if ee_kind == "site":
        mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, ee_id)
        p = mj_data.site_xpos[ee_id].copy()
    else:
        mujoco.mj_jacBody(mj_model, mj_data, jacp, jacr, ee_id)
        p = mj_data.xpos[ee_id].copy()
    return p, jacp

rng = np.random.default_rng(0)
def sample_q():
    q = np.zeros(mj_model.nq)
    for j in range(mj_model.njnt):
        adr = mj_model.jnt_qposadr[j]
        lo, hi = mj_model.jnt_range[j] if mj_model.jnt_limited[j] else (-np.pi, np.pi)
        q[adr] = rng.uniform(lo, hi)
    return q

# ---------- compare over random samples ----------
N, eps = 8, 1e-6
max_p = max_J = max_Jdq = 0.0
for _ in range(N):
    q = sample_q()
    qd = rng.uniform(-1, 1, size=mj_model.nv)

    p_mj, J_mj = mj_pos_and_J(q)
    p_jax = np.asarray(fk_j(jnp.array(q)))
    J_jax = np.asarray(J_j(jnp.array(q)))
    Jdq_jax = np.asarray(jdq_j(jnp.array(q), jnp.array(qd)))

    # finite-diff ground truth for J_dot q_dot: central diff of frame velocity v(q)=J(q) qd
    _, Jp = mj_pos_and_J(q + eps * qd)
    _, Jm = mj_pos_and_J(q - eps * qd)
    Jdq_fd = (Jp @ qd - Jm @ qd) / (2 * eps)

    max_p = max(max_p, np.abs(p_jax - p_mj).max())
    max_J = max(max_J, np.abs(J_jax - J_mj).max())
    max_Jdq = max(max_Jdq, np.abs(Jdq_jax - Jdq_fd).max())

print(f"max |p_jax    - p_mj|     = {max_p:.2e}   (FK position)")
print(f"max |J_jax    - J_mj|     = {max_J:.2e}   (Jacobian vs mj_jac)")
print(f"max |Jdotqd_jvp - FD|     = {max_Jdq:.2e}   (curvature term vs finite diff)")

# ---------- timing of the jitted curvature term ----------
q0, qd0 = jnp.array(sample_q()), jnp.array(rng.uniform(-1, 1, size=mj_model.nv))
jdq_j(q0, qd0).block_until_ready()  # warm compile
M, t0 = 2000, time.perf_counter()
for _ in range(M):
    jdq_j(q0, qd0).block_until_ready()
dt = (time.perf_counter() - t0) / M
print(f"\njitted J_dot q_dot eval: {dt*1e6:.1f} us  (~{1/dt:.0f} Hz, single, CPU)")
