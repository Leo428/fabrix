"""Compare two JAX kinematics providers for single-instance controller latency:

  (A) MJX FK            -- general, loads any MJCF, but built for batched GPU
  (B) lean custom FK    -- serial-chain FK built from the model's joint frames, pure JAX

Both are checked for correctness vs MuJoCo, then timed for fk / J / Jdot*qdot (jitted).
The decision-relevant number is single-call latency (what a controller pays each step).
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

jax.config.update("jax_enable_x64", True)

XML = "/Users/huzheyuan/Documents/kinova/mujoco_menagerie/kinova_gen3/gen3.xml"
mj_model = mujoco.MjModel.from_xml_path(XML)
mj_data = mujoco.MjData(mj_model)
SITE = mj_model.nsite - 1  # 'pinch_site'

# ---------------- MuJoCo ground truth ----------------
def mj_pos_and_J(q):
    mj_data.qpos[:] = q
    mujoco.mj_kinematics(mj_model, mj_data)
    mujoco.mj_comPos(mj_model, mj_data)
    jacp = np.zeros((3, mj_model.nv)); jacr = np.zeros((3, mj_model.nv))
    mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, SITE)
    return mj_data.site_xpos[SITE].copy(), jacp

rng = np.random.default_rng(0)
def sample_q():
    q = np.zeros(mj_model.nq)
    for j in range(mj_model.njnt):
        adr = mj_model.jnt_qposadr[j]
        lo, hi = mj_model.jnt_range[j] if mj_model.jnt_limited[j] else (-np.pi, np.pi)
        q[adr] = rng.uniform(lo, hi)
    return q

# ---------------- (A) MJX provider ----------------
mjx_model = mjx.put_model(mj_model)
def fk_mjx(q):
    d = mjx.make_data(mjx_model).replace(qpos=q)
    d = mjx.kinematics(mjx_model, d)
    return d.site_xpos[SITE]

# ---------------- (B) lean custom serial-chain FK ----------------
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
body_parentid = np.array(mj_model.body_parentid)
body_pos   = jnp.array(mj_model.body_pos)
body_quat  = jnp.array(mj_model.body_quat)
body_jntnum = np.array(mj_model.body_jntnum)
body_jntadr = np.array(mj_model.body_jntadr)
jnt_type   = np.array(mj_model.jnt_type)
jnt_axis   = jnp.array(mj_model.jnt_axis)
jnt_pos    = jnp.array(mj_model.jnt_pos)
jnt_qposadr = np.array(mj_model.jnt_qposadr)
site_bodyid = int(mj_model.site_bodyid[SITE])
site_pos   = jnp.array(mj_model.site_pos[SITE])

def quat_mul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return jnp.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw])

def quat_rot(q, v):
    w, x, y, z = q
    t = 2.0 * jnp.cross(jnp.array([x, y, z]), v)
    return v + w * t + jnp.cross(jnp.array([x, y, z]), t)

def axisangle_quat(axis, angle):
    axis = axis / jnp.linalg.norm(axis)
    h = 0.5 * angle
    return jnp.concatenate([jnp.cos(h)[None], jnp.sin(h) * axis])

def fk_custom(q):
    pw = [jnp.zeros(3)] * mj_model.nbody
    qw = [jnp.array([1., 0., 0., 0.])] * mj_model.nbody
    for b in range(1, mj_model.nbody):
        p = int(body_parentid[b])
        wp = pw[p] + quat_rot(qw[p], body_pos[b])
        wq = quat_mul(qw[p], body_quat[b])
        if body_jntnum[b] == 1 and jnt_type[body_jntadr[b]] == HINGE:
            ja = body_jntadr[b]
            jq = axisangle_quat(jnt_axis[ja], q[jnt_qposadr[ja]])
            anchor = jnt_pos[ja]
            wp = wp + quat_rot(wq, anchor - quat_rot(jq, anchor))
            wq = quat_mul(wq, jq)
        pw[b] = wp; qw[b] = wq
    return pw[site_bodyid] + quat_rot(qw[site_bodyid], site_pos)

# ---------------- helpers: build J and Jdot*qdot for any fk ----------------
def make_ops(fk):
    J = jax.jacobian(fk)
    def jdq(q, qd):
        f1 = lambda qq: jax.jvp(fk, (qq,), (qd,))[1]
        return jax.jvp(f1, (q,), (qd,))[1]
    return jax.jit(fk), jax.jit(J), jax.jit(jdq)

fk_a, J_a, jdq_a = make_ops(fk_mjx)
fk_b, J_b, jdq_b = make_ops(fk_custom)

# ---------------- correctness of custom FK vs MuJoCo ----------------
mp = mj_ = mjd = 0.0
for _ in range(8):
    q = sample_q(); qd = rng.uniform(-1, 1, size=mj_model.nv)
    p_mj, J_mj = mj_pos_and_J(q)
    mp  = max(mp,  np.abs(np.asarray(fk_b(jnp.array(q))) - p_mj).max())
    mj_ = max(mj_, np.abs(np.asarray(J_b(jnp.array(q))) - J_mj).max())
    _, Jp = mj_pos_and_J(q + 1e-6*qd); _, Jm = mj_pos_and_J(q - 1e-6*qd)
    fd = (Jp @ qd - Jm @ qd) / 2e-6
    mjd = max(mjd, np.abs(np.asarray(jdq_b(jnp.array(q), jnp.array(qd))) - fd).max())
print(f"custom FK correctness:  |dp|={mp:.1e}  |dJ|={mj_:.1e}  |dJdotqd|={mjd:.1e}\n")

# ---------------- timing ----------------
def bench(fn, args, M=300):
    fn(*args).block_until_ready()  # warm/compile
    t0 = time.perf_counter()
    for _ in range(M):
        fn(*args).block_until_ready()
    return (time.perf_counter() - t0) / M * 1e6  # us/call

q0 = jnp.array(sample_q()); qd0 = jnp.array(rng.uniform(-1, 1, size=mj_model.nv))
rows = [
    ("MJX",        fk_a, J_a, jdq_a),
    ("custom",     fk_b, J_b, jdq_b),
]
print(f"{'provider':<10}{'fk (us)':>12}{'J (us)':>12}{'Jdotqd (us)':>14}")
for name, fk, J, jdq in rows:
    print(f"{name:<10}{bench(fk,(q0,)):>12.1f}{bench(J,(q0,)):>12.1f}{bench(jdq,(q0,qd0)):>14.1f}")

# bonus: custom FK batched over many configs (the RL/sim-scale path), CPU
B = 4096
qs = jnp.array(np.stack([sample_q() for _ in range(B)]))
qds = jnp.array(rng.uniform(-1, 1, size=(B, mj_model.nv)))
vjdq = jax.jit(jax.vmap(lambda q, qd: (lambda f1: jax.jvp(f1, (q,), (qd,))[1])(
    lambda qq: jax.jvp(fk_custom, (qq,), (qd,))[1])))
vjdq(qs, qds).block_until_ready()
t0 = time.perf_counter()
for _ in range(20):
    vjdq(qs, qds).block_until_ready()
dt = (time.perf_counter() - t0) / 20
print(f"\ncustom FK vmap Jdotqd over {B} configs: {dt*1e3:.1f} ms total "
      f"({dt/B*1e6:.2f} us/config, CPU)")
