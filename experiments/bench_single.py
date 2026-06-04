"""Single-arm (batch=1) latency: what the real-time controller actually pays per step.

Compile is reported ONCE (one-time startup cost), then steady-state per-call latency
(min = best case, median = typical), float32, CPU. No batching -- we control one arm.
"""
import time, statistics
import numpy as np
import jax, jax.numpy as jnp
import mujoco
from mujoco import mjx

XML = "/Users/huzheyuan/Documents/kinova/mujoco_menagerie/kinova_gen3/gen3.xml"
mj_model = mujoco.MjModel.from_xml_path(XML)
SITE = mj_model.nsite - 1

# ---- lean custom serial-chain FK (float32) ----
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
bp = np.array(mj_model.body_parentid)
bpos = jnp.array(mj_model.body_pos, jnp.float32); bq = jnp.array(mj_model.body_quat, jnp.float32)
bjn = np.array(mj_model.body_jntnum); bja = np.array(mj_model.body_jntadr)
jt = np.array(mj_model.jnt_type)
jx = jnp.array(mj_model.jnt_axis, jnp.float32); jp = jnp.array(mj_model.jnt_pos, jnp.float32)
jqa = np.array(mj_model.jnt_qposadr)
sb = int(mj_model.site_bodyid[SITE]); spos = jnp.array(mj_model.site_pos[SITE], jnp.float32)

def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return jnp.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                      aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
def qrot(q, v):
    w, x, y, z = q; u = jnp.array([x, y, z]); t = 2.0*jnp.cross(u, v)
    return v + w*t + jnp.cross(u, t)
def aaq(ax, an):
    ax = ax/jnp.linalg.norm(ax); h = 0.5*an
    return jnp.concatenate([jnp.cos(h)[None], jnp.sin(h)*ax])

def fk(q):
    pw = [jnp.zeros(3, jnp.float32)]*mj_model.nbody
    qw = [jnp.array([1., 0, 0, 0], jnp.float32)]*mj_model.nbody
    for b in range(1, mj_model.nbody):
        p = int(bp[b]); wp = pw[p]+qrot(qw[p], bpos[b]); wq = qmul(qw[p], bq[b])
        if bjn[b] == 1 and jt[bja[b]] == HINGE:
            ja = bja[b]; jq = aaq(jx[ja], q[jqa[ja]]); an = jp[ja]
            wp = wp + qrot(wq, an - qrot(jq, an)); wq = qmul(wq, jq)
        pw[b] = wp; qw[b] = wq
    return pw[sb] + qrot(qw[sb], spos)

# ---- MJX FK for reference ----
mjx_model = mjx.put_model(mj_model); mjx_d0 = mjx.make_data(mjx_model)
def fk_mjx(q):
    return mjx.kinematics(mjx_model, mjx_d0.replace(qpos=q)).site_xpos[SITE]

def jdq(fkf):
    def f(q, qd):
        f1 = lambda qq: jax.jvp(fkf, (qq,), (qd,))[1]
        return jax.jvp(f1, (q,), (qd,))[1]
    return f

q0 = jnp.array(np.random.default_rng(0).uniform(-1, 1, mj_model.nq), jnp.float32)
qd0 = jnp.array(np.random.default_rng(1).uniform(-1, 1, mj_model.nv), jnp.float32)

def measure(jfn, args, budget_s=1.0, max_M=4000):
    t = time.perf_counter(); jfn(*args).block_until_ready()       # 1st call = compile
    comp = (time.perf_counter()-t)*1e3
    ts = []; start = time.perf_counter()
    while len(ts) < max_M and (time.perf_counter()-start) < budget_s:
        t = time.perf_counter(); jfn(*args).block_until_ready(); ts.append(time.perf_counter()-t)
    return comp, min(ts)*1e6, statistics.median(ts)*1e6, len(ts)

ops = [
    ("custom fk",     jax.jit(fk),            (q0,)),
    ("custom Jdotqd", jax.jit(jdq(fk)),       (q0, qd0)),
    ("MJX fk",        jax.jit(fk_mjx),        (q0,)),
    ("MJX Jdotqd",    jax.jit(jdq(fk_mjx)),   (q0, qd0)),
]
print(f"{'op':<16}{'compile (ms, once)':>20}{'min (us)':>12}{'median (us)':>14}{'n':>7}")
for name, jfn, args in ops:
    c, mn, md, n = measure(jfn, args)
    print(f"{name:<16}{c:>20.0f}{mn:>12.1f}{md:>14.1f}{n:>7}")
