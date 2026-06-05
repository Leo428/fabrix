"""Collision-sphere proxies for whole-arm and self-collision avoidance.

A manipulator's links are approximated by a handful of **spheres** rigidly fixed in their link
frames (the NVIDIA-fabrics / cuRobo idea). Two kinds of barrier then keep the arm safe:

- **self-collision** — a barrier on the distance between each *non-adjacent* sphere pair, so the
  arm cannot fold into itself;
- **whole-arm environment** — a barrier on the distance from each sphere to an obstacle / floor, so
  the *elbow and forearm* (not just the end-effector) avoid the world.

All of them are **one batched leaf**, not one-leaf-per-pair: a single FK to every link frame
(:meth:`fabrix.kinematics.CustomFK.body_poses`) places all spheres, the pairwise/region distances
are vectorized, and the ``k`` per-distance barriers are summed in closed form by the generalized
:func:`fabrix.geometry.sdf_barrier_geometry` / :func:`~fabrix.geometry.sdf_barrier_potential`. Profiling
showed this is bit-identical to ``k`` separate leaves but ~flat in compile time and runtime (separate
leaves were 41 s to compile at k=64; batched 2.7 s), which is what makes ~tens of pairs practical.

Pair a geometry (energized, speed-conserving deflection) with a potential (diverging at contact, the
hard non-collision invariant) — the same M2 division of labour as the obstacle/joint barriers.

The sphere model is generated **automatically** from the kinematics (:func:`auto_arm_spheres`) and can
then be **hand-tuned** (edit a :class:`SphereModel`, or author one with :meth:`SphereModel.from_dict`).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from fabrix.geometry import sdf_barrier_geometry, sdf_barrier_potential
from fabrix.kinematics import _qrot
from fabrix.spec import dynamic_gain

_SOFT = 1e-6  # softens the pairwise norm so its gradient stays finite if two centers coincide


def _body_id(m, name: str) -> int:
    """Resolve a body name to its id, raising a clear error if it is unknown."""
    import mujoco
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"unknown body {name!r}")
    return int(bid)


@dataclasses.dataclass(frozen=True)
class SphereModel:
    """Collision spheres rigidly attached to link frames (a *static*, untraced description).

    Fields are parallel arrays over the ``Ns`` spheres: ``link`` (body id each sphere rides on),
    ``local`` (``(Ns, 3)`` center in that body's frame), ``radius`` (``(Ns,)``). Build one with
    :func:`auto_arm_spheres`, hand-author with :meth:`from_dict`, or edit the arrays directly — the
    barrier leaves just read them at construction time.
    """

    link: np.ndarray      # (Ns,)  body id per sphere
    local: np.ndarray     # (Ns,3) center in the body frame
    radius: np.ndarray    # (Ns,)  sphere radius

    def __len__(self) -> int:
        return int(len(self.link))

    @classmethod
    def from_dict(cls, provider, spec: Dict[str, List[Tuple]]) -> "SphereModel":
        """Author spheres by body name: ``{"forearm_link": [((x, y, z), r), ...], ...}``.

        ``(x, y, z)`` is the center in that body's frame, ``r`` the radius. The entry point for
        hand-tuning: start from :func:`auto_arm_spheres`, then refine specific links here.
        """
        m = provider.mj_model
        link, local, radius = [], [], []
        for name, spheres in spec.items():
            bid = _body_id(m, name)
            for center, r in spheres:
                link.append(bid); local.append(center); radius.append(r)
        return cls(np.array(link, int), np.array(local, float).reshape(-1, 3), np.array(radius, float))

    def to_dict(self, provider, decimals: int = 6) -> Dict[str, List[Tuple]]:
        """Dump to the ``from_dict`` format ``{body_name: [((x, y, z), r), ...]}`` (inverse of
        :meth:`from_dict`). The committable hand-tuning artifact: export an auto model, edit, reload.

        Spheres are grouped by body in index order; values are rounded to ``decimals`` for a readable,
        diffable literal (6 = micron precision, so the round-trip is lossless at hardware tolerances).
        """
        names = self.names(provider)
        out: Dict[str, List[Tuple]] = {}
        for i, name in enumerate(names):
            out.setdefault(name, []).append(
                (tuple(round(float(x), decimals) for x in self.local[i]), round(float(self.radius[i]), decimals)))
        return out

    def scaled(self, provider, factors: Dict[str, float]) -> "SphereModel":
        """Return a copy with per-link radius multipliers applied (``{body_name: factor}``).

        Links not listed keep their radius. The one-liner for the common "auto over-covers a few links"
        fix — e.g. ``sph.scaled(prov, {"forearm_link": 0.7})``. ``local`` and ``link`` are unchanged.
        """
        m = provider.mj_model
        fac = np.ones(len(self))
        for name, f in factors.items():
            fac[self.link == _body_id(m, name)] = float(f)
        return SphereModel(self.link.copy(), self.local.copy(), self.radius * fac)

    def names(self, provider) -> List[str]:
        """Body name each sphere rides on (for inspecting/tuning an auto-generated model)."""
        import mujoco
        return [mujoco.mj_id2name(provider.mj_model, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in self.link]


def _collision_radius(m, b: int, default: float = 0.05) -> float:
    """Cross-section radius for body ``b`` from its collision geom's bounding width (``max(size[:2])``)."""
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) == b and (m.geom_contype[g] or m.geom_conaffinity[g]):
            return float(max(m.geom_size[g][0], m.geom_size[g][1]))
    return default


def _child_offset(m, b: int) -> Optional[np.ndarray]:
    """The 'bone' vector of link ``b`` in its own frame: the offset to its child joint, or, for the
    leaf link carrying the tracked site, the offset to that site. ``None`` if neither exists."""
    children = np.flatnonzero(m.body_parentid == b)
    children = children[children != b]
    if len(children):
        return np.asarray(m.body_pos[children[0]], float)   # serial chain: one child
    site = m.nsite - 1
    if int(m.site_bodyid[site]) == b:
        return np.asarray(m.site_pos[site], float)
    return None


def auto_arm_spheres(provider, n_per_link: int = 2, radius: Optional[float] = None,
                     radius_scale: Optional[Dict[str, float]] = None) -> SphereModel:
    """Auto-place ``n_per_link`` spheres along each link's 'bone' (the segment toward its child joint).

    For every movable link, spheres sit at interior fractions of the bone in the link frame (so they
    cover the link body without piling up on the shared joints), with ``radius`` taken from the link's
    collision-mesh cross-section unless one is given. Reproducible and model-general; the batched leaves
    make the sphere count nearly free, so prefer over-covering. Hand-tune the result if a link needs it.

    ``radius_scale`` (``{body_name: factor}``) multiplies the per-link radius for named links only — the
    quick fix when the mesh cross-section over-covers a few links. Equivalent to
    :meth:`SphereModel.scaled` on the result; for finer edits (move/add/drop) use the tuner + ``from_dict``.
    """
    m = provider.mj_model
    scale = {_body_id(m, n): float(f) for n, f in (radius_scale or {}).items()}
    fracs = np.linspace(0.0, 1.0, n_per_link + 2)[1:-1]     # interior points, off the joints
    link, local, rad = [], [], []
    for b in range(1, m.nbody):                              # skip world (0)
        bone = _child_offset(m, b)
        if bone is None:
            continue
        r = (radius if radius is not None else _collision_radius(m, b)) * scale.get(b, 1.0)
        for t in fracs:
            link.append(b); local.append(t * bone); rad.append(r)
    return SphereModel(np.array(link, int), np.array(local, float).reshape(-1, 3), np.array(rad, float))


def nonadjacent_pairs(spheres: SphereModel, provider) -> np.ndarray:
    """Sphere index pairs ``(Npairs, 2)`` to check for self-collision: distinct, **non-adjacent** links.

    Excludes pairs on the same link and on parent-child links (which meet at a joint and whose spheres
    necessarily overlap there). Far-apart pairs are kept — they cost almost nothing in the batched leaf
    and their barriers are inert until the links actually approach.
    """
    parent = provider.mj_model.body_parentid
    L = spheres.link
    pairs = [(a, b) for a in range(len(L)) for b in range(a + 1, len(L))
             if L[a] != L[b] and parent[L[a]] != L[b] and parent[L[b]] != L[a]]
    return np.array(pairs, int).reshape(-1, 2)


def load_spheres(provider, path, n_per_link: int = 2, **auto_kwargs) -> Tuple[SphereModel, bool]:
    """Load hand-tuned spheres from ``path`` if it exists, else fall back to :func:`auto_arm_spheres`.

    ``path`` is a Python module (e.g. one written by ``demos/tune_spheres.py``) exposing either a
    ``load_tuned(provider)`` function or a ``SPHERES`` dict in :meth:`SphereModel.from_dict` form.
    Returns ``(model, is_tuned)`` so callers can report which source was used. This is the runtime end
    of the "auto first, hand-tune if needed" pipeline — a demo just calls it and uses whatever it gets.
    """
    import importlib.util
    import pathlib
    p = pathlib.Path(path)
    if p.exists():
        spec = importlib.util.spec_from_file_location("_fabrix_tuned_spheres", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        model = mod.load_tuned(provider) if hasattr(mod, "load_tuned") \
            else SphereModel.from_dict(provider, mod.SPHERES)
        return model, True
    return auto_arm_spheres(provider, n_per_link=n_per_link, **auto_kwargs), False


# ---------------------------------------------------------------------------
# Distance fields over the sphere set (each returns (k,), fed to the batched barrier core).
# ---------------------------------------------------------------------------
def _centers(provider, link, local):
    """World centers ``(Ns, 3)`` of the spheres for configuration ``q`` (one shared FK to all links)."""
    def fn(q):
        P, Q = provider.body_poses(q)                        # (nbody,3), (nbody,4)
        return P[link] + jax.vmap(_qrot)(Q[link], local)     # gather link frames + rotate local offsets
    return fn


def _arrays(spheres: SphereModel):
    return jnp.asarray(spheres.link), jnp.asarray(spheres.local), jnp.asarray(spheres.radius)


def self_collision_geometry(provider, spheres: SphereModel, pairs: np.ndarray, k_b: float = 1.0,
                            power: float = 2.0, m_b: float = 2.0, d0: Optional[float] = 0.05,
                            margin: float = 0.0, eps: float = 1e-3):
    """Energized barrier *geometry* on every non-adjacent sphere-pair gap — deflects the arm from itself."""
    link, local, rad = _arrays(spheres)
    pa, pb = jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])
    rsum = rad[pa] + rad[pb]
    centers = _centers(provider, link, local)

    def dist(q, params):
        c = centers(q)
        diff = c[pa] - c[pb]
        # astype: sphere constants may be float64; anchor to the config dtype (M3 promotion guard,
        # a no-op in float32 deployment) so a float32 scan carry stays float32.
        return (jnp.sqrt(jnp.sum(diff * diff, axis=1) + _SOFT * _SOFT) - rsum).astype(q.dtype)  # (Npairs,)

    return sdf_barrier_geometry(dist, k_b=k_b, power=power, m_b=m_b, d0=d0, margin=margin, eps=eps)


def self_collision_potential(provider, spheres: SphereModel, pairs: np.ndarray, k_p: float = 0.1,
                             d0: float = 0.03, m_p: float = 2.0, margin: float = 0.0, eps: float = 1e-3):
    """Diverging barrier *potential* on every non-adjacent sphere-pair gap — the hard no-self-collision wall."""
    link, local, rad = _arrays(spheres)
    pa, pb = jnp.asarray(pairs[:, 0]), jnp.asarray(pairs[:, 1])
    rsum = rad[pa] + rad[pb]
    centers = _centers(provider, link, local)

    def dist(q, params):
        c = centers(q)
        diff = c[pa] - c[pb]
        return (jnp.sqrt(jnp.sum(diff * diff, axis=1) + _SOFT * _SOFT) - rsum).astype(q.dtype)

    return sdf_barrier_potential(dist, k_p=k_p, d0=d0, m_p=m_p, margin=margin, eps=eps)


def arm_obstacle_geometry(provider, spheres: SphereModel, center, radius: float, k_b: float = 1.0,
                          power: float = 2.0, m_b: float = 2.0, d0: Optional[float] = 0.12,
                          margin: float = 0.0, eps: float = 1e-3):
    """Energized barrier geometry keeping **every** arm sphere outside a sphere obstacle.

    The whole-arm counterpart of :func:`fabrix.geometry.obstacle_geometry` (which guards only the EE).
    ``center=None`` reads ``params.obstacle_center`` (a draggable / moving obstacle).
    """
    link, local, rad = _arrays(spheres)
    centers = _centers(provider, link, local)
    cfix = None if center is None else jnp.asarray(center)

    def dist(q, params):
        c = centers(q)
        o = params.obstacle_center if cfix is None else cfix
        diff = c - o
        return (jnp.sqrt(jnp.sum(diff * diff, axis=1) + _SOFT * _SOFT)
                - (rad + dynamic_gain(radius, params))).astype(q.dtype)  # (Ns,)

    return sdf_barrier_geometry(dist, k_b=k_b, power=power, m_b=m_b, d0=d0, margin=margin, eps=eps)


def arm_obstacle_potential(provider, spheres: SphereModel, center, radius: float, k_p: float = 0.5,
                           d0: float = 0.02, m_p: float = 4.0, margin: float = 0.0, eps: float = 1e-3):
    """Diverging barrier potential keeping every arm sphere outside a sphere obstacle (the hard wall)."""
    link, local, rad = _arrays(spheres)
    centers = _centers(provider, link, local)
    cfix = None if center is None else jnp.asarray(center)

    def dist(q, params):
        c = centers(q)
        o = params.obstacle_center if cfix is None else cfix
        diff = c - o
        return (jnp.sqrt(jnp.sum(diff * diff, axis=1) + _SOFT * _SOFT)
                - (rad + dynamic_gain(radius, params))).astype(q.dtype)

    return sdf_barrier_potential(dist, k_p=k_p, d0=d0, m_p=m_p, margin=margin, eps=eps)


def arm_plane_geometry(provider, spheres: SphereModel, point, normal, k_b: float = 1.0,
                       power: float = 2.0, m_b: float = 2.0, d0: Optional[float] = 0.12,
                       margin: float = 0.0, eps: float = 1e-3):
    """Energized barrier geometry keeping every arm sphere on the ``+normal`` side of a plane (e.g. a floor)."""
    link, local, rad = _arrays(spheres)
    centers = _centers(provider, link, local)
    p0 = jnp.asarray(point)
    n = jnp.asarray(normal)
    n = n / jnp.linalg.norm(n)

    def dist(q, params):
        return ((centers(q) - p0) @ n - rad).astype(q.dtype)   # (Ns,) each sphere's signed clearance

    return sdf_barrier_geometry(dist, k_b=k_b, power=power, m_b=m_b, d0=d0, margin=margin, eps=eps)


def arm_plane_potential(provider, spheres: SphereModel, point, normal, k_p: float = 0.5,
                        d0: float = 0.05, m_p: float = 4.0, margin: float = 0.0, eps: float = 1e-3):
    """Diverging barrier potential keeping every arm sphere on the ``+normal`` side of a plane."""
    link, local, rad = _arrays(spheres)
    centers = _centers(provider, link, local)
    p0 = jnp.asarray(point)
    n = jnp.asarray(normal)
    n = n / jnp.linalg.norm(n)

    def dist(q, params):
        return ((centers(q) - p0) @ n - rad).astype(q.dtype)

    return sdf_barrier_potential(dist, k_p=k_p, d0=d0, m_p=m_p, margin=margin, eps=eps)
