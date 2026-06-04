"""Clean latency measurement (float32, no x64).

Reports MIN latency (robust to load spikes), separates:
  - per-call latency  : one Python dispatch + compute + block  (what a controller pays/step)
  - compute-only      : N evals fused inside one lax.scan dispatch, divided by N
"""
import os, time, statistics
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
import mujoco
from mujoco import mjx

print("ncpu:", os.cpu_count(), "| loadavg:", [round(x, 2) for x in os.getloadavg()])
print("x64:", jax.config.jax_enable_x64, "| devices:", jax.devices(), "\n")

XML = "/Users/huzheyuan/Documents/kinova/mujoco_menagerie/kinova_gen3/gen3.xml"
mj_model = mujoco.MjModel.from_xml_path(XML)
SITE = mj_model.nsite - 1
nv = mj_model.nv

# ---- lean custom serial-chain FK (float32) ----
HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
bp  = np.array(mj_model.body_parentid)
bpos = jnp.array(mj_model.body_pos, jnp.float32)
bq   = jnp.array(mj_model.body_quat, jnp.float32)
bjn  = np.array(mj_model.body_jntnum); bja = np.array(mj_model.body_jntadr)
jt   = np.array(mj_model.jnt_type)
jax_ = jnp.array(mj_model.jnt_axis, jnp.float32)
jpos = jnp.array(mj_model.jnt_pos, jnp.float32)
jqa  = np.array(mj_model.jnt_qposadr)
sb   = int(mj_model.site_bodyid[SITE]); spos = jnp.array(mj_model.site_pos[SITE], jnp.float32)

def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return jnp.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                      aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])
def qrot(q, v):
    w, x, y, z = q; u = jnp.array([x, y, z]); t = 2.0*jnp.cross(u, v)
    return v + w*t + jnp.cross(u, t)
def aaq(axis, ang):
    axis = axis/jnp.linalg.norm(axis); h = 0.5*ang
    return jnp.concatenate([jnp.cos(h)[None], jnp.sin(h)*axis])

def fk(q):
    pw = [jnp.zeros(3, jnp.float32)]*mj_model.nbody
    qw = [jnp.array([1., 0, 0, 0], jnp.float32)]*mj_model.nbody
    for b in range(1, mj_model.nbody):
        p = int(bp[b]); wp = pw[p]+qrot(qw[p], bpos[b]); wq = qmul(qw[p], bq[b])
        if bjn[b] == 1 and jt[bja[b]] == HINGE:
            ja = bja[b]; jq = aaq(jax_[ja], q[jqa[ja]]); an = jpos[ja]
            wp = wp + qrot(wq, an - qrot(jq, an)); wq = qmul(wq, jq)
        pw[b] = wp; qw[b] = wq
    return pw[sb] + qrot(qw[sb], spos)

def jdq_fn(q, qd):
    f1 = lambda qq: jax.jvp(fk, (qq,), (qd,))[1]
    return jax.jvp(f1, (q,), (qd,))[1]

# ---- MJX provider for reference ----
mjx_model = mjx.put_model(mj_model)
mjx_d0 = mjx.make_data(mjx_model)
def fk_mjx(q):
    d = mjx.kinematics(mjx_model, mjx_d0.replace(qpos=q))
    return d.site_xpos[SITE]
def jdq_mjx(q, qd):
    f1 = lambda qq: jax.jvp(fk_mjx, (qq,), (qd,))[1]
    return jax.jvp(f1, (q,), (qd,))[1]

q0  = jnp.array(np.random.default_rng(0).uniform(-1, 1, mj_model.nq), jnp.float32)
qd0 = jnp.array(np.random.default_rng(1).uniform(-1, 1, nv), jnp.float32)

def per_call_min(fn, args, M=3000):
    fn(*args).block_until_ready()
    ts = []
    for _ in range(M):
        t = time.perf_counter(); fn(*args).block_until_ready(); ts.append(time.perf_counter()-t)
    return min(ts)*1e6, statistics.median(ts)*1e6

def compute_only(raw_fn, args, N=2000):
    """N fused evals in one dispatch (input perturbed to defeat CSE)."""
    def scanned(*a):
        q, rest = a[0], a[1:]
        def step(c, i):
            return c, raw_fn(q + 1e-4*jnp.sin(i.astype(jnp.float32)), *rest)
        return lax.scan(step, None, jnp.arange(N))[1]
    j = jax.jit(scanned)
    j(*args).block_until_ready()
    t = time.perf_counter(); j(*args).block_until_ready(); dt = time.perf_counter()-t
    return dt/N*1e6

fk_j   = jax.jit(fk);      jdq_j   = jax.jit(jdq_fn)
fk_mj  = jax.jit(fk_mjx);  jdq_mj  = jax.jit(jdq_mjx)

print(f"{'op':<22}{'per-call min (us)':>20}{'median (us)':>14}{'compute-only (us)':>20}")
for name, jfn, raw, args in [
    ("custom fk",       fk_j,  fk,      (q0,)),
    ("custom Jdotqd",   jdq_j, jdq_fn,  (q0, qd0)),
    ("MJX fk",          fk_mj, fk_mjx,  (q0,)),
    ("MJX Jdotqd",      jdq_mj, jdq_mjx,(q0, qd0)),
]:
    mn, md = per_call_min(jfn, args)
    co = compute_only(raw, args)
    print(f"{name:<22}{mn:>20.1f}{md:>14.1f}{co:>20.2f}")

# batched custom Jdotqd (RL/sim-scale), compute-only per config, min of a few
print()
for B in (64, 1024, 8192):
    qs  = jnp.array(np.random.default_rng(2).uniform(-1, 1, (B, mj_model.nq)), jnp.float32)
    qds = jnp.array(np.random.default_rng(3).uniform(-1, 1, (B, nv)), jnp.float32)
    vf = jax.jit(jax.vmap(jdq_fn))
    vf(qs, qds).block_until_ready()
    ts = []
    for _ in range(30):
        t = time.perf_counter(); vf(qs, qds).block_until_ready(); ts.append(time.perf_counter()-t)
    print(f"custom Jdotqd vmap B={B:<6} : {min(ts)*1e3:7.2f} ms/batch  ({min(ts)/B*1e6:6.2f} us/config)")
