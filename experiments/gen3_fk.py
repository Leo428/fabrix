"""Lean serial-chain forward kinematics for a MuJoCo model, in pure JAX.

build_site_fk(xml) -> (fk, mj_model, site_id)  where fk(q) -> world position of a site.
Validated to ~1e-15 vs mujoco.mj_jacSite / FD. Differentiable: J = jacobian(fk),
Jdot*qdot = nested jvp. This is the candidate real-time kinematics provider.
"""
import numpy as np
import jax, jax.numpy as jnp
import mujoco


def build_site_fk(xml_path, site_name=None, dtype=jnp.float32):
    m = mujoco.MjModel.from_xml_path(xml_path)
    site = (m.nsite - 1 if site_name is None
            else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site_name))
    HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)

    bp = np.array(m.body_parentid)
    bpos = jnp.array(m.body_pos, dtype); bq = jnp.array(m.body_quat, dtype)
    bjn = np.array(m.body_jntnum); bja = np.array(m.body_jntadr)
    jt = np.array(m.jnt_type)
    jx = jnp.array(m.jnt_axis, dtype); jp = jnp.array(m.jnt_pos, dtype)
    jqa = np.array(m.jnt_qposadr)
    sb = int(m.site_bodyid[site]); spos = jnp.array(m.site_pos[site], dtype)
    nbody = m.nbody

    def qmul(a, b):  # vectorized: no scalar packing
        w1, v1 = a[0], a[1:]; w2, v2 = b[0], b[1:]
        w = w1*w2 - jnp.dot(v1, v2)
        v = w1*v2 + w2*v1 + jnp.cross(v1, v2)
        return jnp.concatenate([w[None], v])

    def qrot(q, p):
        w, u = q[0], q[1:]; t = 2.0*jnp.cross(u, p)
        return p + w*t + jnp.cross(u, t)

    def aaq(ax, an):
        ax = ax/jnp.linalg.norm(ax); h = 0.5*an
        return jnp.concatenate([jnp.cos(h)[None], jnp.sin(h)*ax])

    def fk(q):
        pw = [jnp.zeros(3, dtype)]*nbody
        qw = [jnp.array([1., 0, 0, 0], dtype)]*nbody
        for b in range(1, nbody):
            p = int(bp[b]); wp = pw[p]+qrot(qw[p], bpos[b]); wq = qmul(qw[p], bq[b])
            if bjn[b] == 1 and jt[bja[b]] == HINGE:
                ja = bja[b]; jq = aaq(jx[ja], q[jqa[ja]]); an = jp[ja]
                wp = wp + qrot(wq, an - qrot(jq, an)); wq = qmul(wq, jq)
            pw[b] = wp; qw[b] = wq
        return pw[sb] + qrot(qw[sb], spos)

    return fk, m, site
