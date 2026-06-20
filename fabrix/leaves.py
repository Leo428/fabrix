"""Leaf factories: each returns ``leaf(q, qd, params) -> Spec`` in configuration space.

Every leaf follows one pattern: define a **desired acceleration** ``a_des`` and a **priority
metric** ``M``, and emit force ``f = -M @ a_des`` so the leaf's *isolated* acceleration is
exactly ``a_des`` while ``M`` sets how strongly the leaf wins the metric-weighted combination.
This is the RMP/fabric design: the task attractor carries a large metric so it dominates in the
task directions, while posture/damping act where the task metric is small (the nullspace) —
so redundancy is resolved without biasing the end-effector equilibrium.

Task-space leaves (``attractor``) are pulled back to configuration space; config-space leaves
(``posture``, ``config_damping``) are identity-map leaves returned directly. M1 leaves are
*forced* (metric + potential-gradient + damping); HD2 geometries + energization arrive in M2.
Gains are baked at construction (mink-style); ``params.target``/``params.q_default`` are traced.
"""
from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp

from fabrix.diff import value_jac_curv
from fabrix.maps import control_points_error_map, se3_pose_error_map, site_position_map
from fabrix.spec import Spec, dynamic_gain, pullback


def _restoring(e, k: float, f_max: Optional[float], eps: float = 1e-3):
    """Restoring 'force' of an attractor potential as a function of the task error ``e``.

    ``f_max=None`` → quadratic potential: gradient ``k·e`` (force grows without bound, so a far
    target commands a large acceleration). Otherwise a **gradient-saturating** potential: magnitude
    ``k·‖e‖`` near the goal (same stiffness/feel as the quadratic) but capped at ``f_max`` far away,
    so the commanded acceleration stays bounded no matter how distant the target — gentle, lunge-free
    large moves. ``f(e) = f_max·tanh(k‖e‖/f_max)·ê``; the softened norm keeps it smooth at ``e=0``.
    """
    if f_max is None:
        return k * e
    r = jnp.sqrt(e @ e + eps * eps)              # softened norm: smooth + NaN-free at e=0
    return f_max * jnp.tanh(k * r / f_max) * (e / r)


def _scaled_mass(e, m_min, m_max, sharp: float, offset: float, eps: float = 1e-3):
    """Distance-scaled attractor priority metric ``m(‖e‖)``: ``m_max`` near the goal, ``m_min`` far.

    NVlabs-style tanh switch ``(m_max−m_min)·(½tanh(−sharp·(r−offset))+½) + m_min`` on the softened
    norm ``r=√(e·e+eps²)`` (smooth + NaN-free at ``e=0``). A high near-goal mass makes the attractor
    dominate the metric-weighted combine so a low-priority posture leaf cannot bias the equilibrium
    (kills the standing TCP offset + slow orbit); the low far-field mass keeps large moves gentle and
    lets posture/geometry lead. ``m_max is None`` → constant ``m_min`` (the M1–M3 behavior, exact).

    ``m`` scales BOTH the metric and the force ``f = m·(restoring+b·ẋ)`` equally, so it cancels from the
    isolated accel ``a_des`` — the schedule only re-weights combine priority, never the convergence
    dynamics (``k``, ``b``) or the saturating ``f_max``.
    """
    if m_max is None:
        return m_min
    r = jnp.sqrt(e @ e + eps * eps)
    s = 0.5 * jnp.tanh(-sharp * (r - offset)) + 0.5
    return (m_max - m_min) * s + m_min


def attractor(provider, k: float = 16.0, b: float = 8.0, m: float = 50.0,
              f_max: Optional[float] = None, m_max=None, sharp: float = 10.0, offset: float = 0.1):
    """EE-position attractor. Desired accel ``-(g(x-x*) + b ẋ)``; priority metric ``m(‖e‖)·I₃``.

    ``k`` sets stiffness (ω=√k), ``b`` damping (b=2√k is critical), ``m`` the task priority
    relative to posture/damping. ``g`` is the restoring force: quadratic (``k·e``) by default, or
    gradient-saturating with magnitude capped at ``f_max`` (see :func:`_restoring`).

    With ``m_max`` set, the priority metric is **distance-scaled** ``m(‖e‖)`` (see :func:`_scaled_mass`):
    ``m`` is the far-field floor and ``m_max`` the near-goal priority, switching at radius ``offset``
    with sharpness ``sharp``. High near the goal → the attractor dominates so a posture leaf cannot bias
    the EE equilibrium; low far away → gentle large moves. ``m_max=None`` (default) → the constant ``m``
    (unchanged M1–M3 behavior).
    """
    phi = site_position_map(provider)

    def leaf(q, qd, params):
        x, J, Jdq = value_jac_curv(phi, q, qd)
        e = x - params.target
        xd = J @ qd
        k_, b_, fm_, mmax_ = (dynamic_gain(g, params) for g in (k, b, f_max, m_max))
        m_min_, sh_, of_ = (dynamic_gain(g, params) for g in (m, sharp, offset))
        m_ = _scaled_mass(e, m_min_, mmax_, sh_, of_)
        M = m_ * jnp.eye(3, dtype=x.dtype)
        f = m_ * (_restoring(e, k_, fm_) + b_ * xd)  # = -M @ a_des
        return pullback(Spec(M, f), J, Jdq)

    return leaf


def pose_attractor(provider, k: float = 16.0, b: float = 8.0, m: float = 50.0,
                   f_max: Optional[float] = None, m_max=None, sharp: float = 10.0, offset: float = 0.1):
    """Full 6-DOF SE(3) pose attractor. Drives the coupled pose error to zero.

    Task = ``e(q) = Log(T*^{-1} T(q)) in se(3)`` (``params.target`` position + ``params.target_quat``
    orientation, wxyz). Desired accel ``-(g(e) + b ė)``; priority metric ``m(‖e‖)·I₆`` (one shared metric
    over the 6 twist coordinates, the coupled-SE(3) choice). ``g`` is the restoring force: quadratic
    by default, or gradient-saturating capped at ``f_max`` (bounded accel on far/commanded moves; note
    one ``f_max`` mixes the translation (m) and rotation (rad) scales of the 6-D twist). Use alongside
    :func:`posture`/:func:`config_damping` to resolve redundancy; the approach is a geodesic screw.

    With ``m_max`` set, the metric is **distance-scaled** ``m(‖e‖)`` (see :func:`_scaled_mass`): high near
    the goal so the attractor dominates a posture leaf (no EE-equilibrium bias / orbit), low far away for
    gentle large moves. ``offset`` (the switch radius) is on the same mixed-units 6-D ``‖e‖`` as ``f_max``.
    ``m_max=None`` (default) → the constant ``m`` (unchanged M1–M3 behavior).
    """

    def leaf(q, qd, params):
        phi = se3_pose_error_map(provider, params.target, params.target_quat)
        e, J, Jdq = value_jac_curv(phi, q, qd)
        ed = J @ qd
        k_, b_, fm_, mmax_ = (dynamic_gain(g, params) for g in (k, b, f_max, m_max))
        m_min_, sh_, of_ = (dynamic_gain(g, params) for g in (m, sharp, offset))
        m_ = _scaled_mass(e, m_min_, mmax_, sh_, of_)
        M = m_ * jnp.eye(6, dtype=e.dtype)
        f = m_ * (_restoring(e, k_, fm_) + b_ * ed)  # = -M @ a_des
        return pullback(Spec(M, f), J, Jdq)

    return leaf


def pose_points_attractor(provider, offsets, k: float = 16.0, b: float = 8.0, m: float = 50.0,
                          f_max: Optional[float] = None, m_max=None, sharp: float = 10.0,
                          offset: float = 0.1):
    """Full 6-DOF pose attractor via rigid CONTROL POINTS — the NVlabs/FABRICS construction.

    A drop-in alternative to :func:`pose_attractor` that avoids the SE(3) twist entirely. Tracks ``P≥3``
    rigid points on the EE (``offsets``: a ``(P,3)`` local jack, e.g. :func:`maps.control_point_jack`)
    via :func:`maps.control_points_error_map` — a purely **positional** attractor over the stacked
    ``(3P,)`` error, ALL in meters. Desired accel ``-(g(e) + b·ė)``; priority metric ``m(‖e‖)·I₃ₚ``.

    Why this over the coupled twist (:func:`pose_attractor`): one uniform metric/gain/``f_max``/``offset``
    over consistent (meter) units — no mixed (m, rad) scales — so the rotation/translation speed balance
    becomes the geometric point radius ``r`` of the jack (in saturation ``v_trans ≈ f_max/b`` is
    ``r``-independent while ``ω_rot ≈ f_max/(b·r)``, so the two are tuned independently) instead of a
    single shared twist cap that throttled rotation to ``f_max/b`` rad/s. ``g`` is the restoring force
    (quadratic, or gradient-saturating capped at ``f_max``); ``m_max`` enables the distance-scaled mass
    (:func:`_scaled_mass`) on the now pure-meters ``‖e‖`` (cleaner than the twist's mixed-units ``‖e‖``).
    Reuses the position-attractor machinery unchanged. Pair with :func:`posture`/:func:`speed_control` to
    resolve redundancy; ``m_max=None`` (default) → the constant ``m``.
    """
    offs = jnp.asarray(offsets)

    def leaf(q, qd, params):
        phi = control_points_error_map(provider, params.target, params.target_quat, offs)
        e, J, Jdq = value_jac_curv(phi, q, qd)
        ed = J @ qd
        k_, b_, fm_, mmax_ = (dynamic_gain(g, params) for g in (k, b, f_max, m_max))
        m_min_, sh_, of_ = (dynamic_gain(g, params) for g in (m, sharp, offset))
        m_ = _scaled_mass(e, m_min_, mmax_, sh_, of_)
        M = m_ * jnp.eye(e.shape[0], dtype=e.dtype)
        f = m_ * (_restoring(e, k_, fm_) + b_ * ed)  # = -M @ a_des
        return pullback(Spec(M, f), J, Jdq)

    return leaf


def posture(nq: int, k=1.0, b: float = 2.0, weight=0.5):
    """Config-space attractor toward ``params.q_default``; low priority → acts in the nullspace.

    Resolves the arm's redundancy toward a nominal (e.g. upright/compact) posture without biasing the
    EE. ``weight`` and ``k`` may be scalars **or per-joint ``(nq,)`` arrays** — use a per-joint
    ``weight`` to hold the uprightness-critical joints (shoulder/elbow) firmly toward ``q_default``
    while leaving the wrist free for the task. Per-joint ``weight`` cancels from the isolated
    acceleration, so it only re-weights priority (which joints win the spare DOF), never the target.

    Ceiling worth knowing: posture acts only in the *task nullspace*. A full 6-DOF pose task leaves a
    7-DOF arm just 1 spare DOF (the elbow swivel), so posture's reach there is small; a position-only
    (or low-orientation-weight) task frees 4 DOF and posture becomes far more effective. Hard
    "never enter this pose" guarantees come from joint-limit barriers, not posture.
    """
    def leaf(q, qd, params):
        w = jnp.asarray(dynamic_gain(weight, params))
        kk = jnp.asarray(dynamic_gain(k, params))
        b_ = dynamic_gain(b, params)
        wv = jnp.broadcast_to(w, (nq,)).astype(q.dtype)
        M = jnp.diag(wv)
        f = wv * (kk * (q - params.q_default) + b_ * qd)  # = -M @ a_des (per-joint weight cancels in a_des)
        return Spec(M, f)

    return leaf


def cspace_attractor(nq: int, gain=1.0, sharp: float = 10.0, weight=1.0, eps: float = 1e-3):
    """NVlabs-style config-space attractor toward ``params.q_default`` as an ENERGIZED HD2 GEOMETRY.

    The faithful port of fabrics_sim's ``cspace_attractor`` (which is a *geometry*, not a forced
    potential like :func:`posture`). Isolated acceleration::

        e = q - params.q_default,   r = ‖e‖ (softened),   ê = e / r
        a = -‖q̇‖² · gain·tanh(sharp·r) · ê                 # points toward q_default

    Three properties distinguish it from the linear-spring :func:`posture` (and they map 1:1 onto the
    "feels asymmetric" complaint the spring caused):

    * **Saturating conical pull** — magnitude ``gain·tanh(sharp·r)``: ~linear near home (slope
      ``gain·sharp`` → a crisp return) but **capped at ``gain`` far away**, so a big excursion for a
      dexterous move is *not* fought by an ever-growing force (the spring's ``k·r`` is). ``gain`` caps
      the far-field resistance; ``sharp`` sets the near-home snap — two independent knobs the single
      spring ``k`` could not separate.
    * **Homogeneous degree 2** (the ``‖q̇‖²`` factor) — a true geometry: **zero force at rest**, paths
      independent of execution speed. It shapes motion already underway rather than statically yanking,
      so it never competes for the "should I be moving" decision — energy + damping own that.
    * **No internal damping** — put this in ``GeometricFabric.geometries`` so it is root-energized
      against the execution energy; convergence/dissipation come from the separate :func:`speed_control`
      + reference damping (NVlabs' decoupled cspace energy + damper).

    Priority metric is the constant isotropic ``weight·I`` (their ``isotropic_metric``); ``weight`` may
    be a per-joint ``(nq,)`` array (hold the shoulder/elbow toward home, free the wrist) and cancels
    from ``a``. ``weight=0`` ⇒ ``M=0, f=0`` ⇒ a fully inert leaf (exact no-op — the wired default).
    """
    def leaf(q, qd, params):
        g_, sh_, w_ = (dynamic_gain(x, params) for x in (gain, sharp, weight))
        wv = jnp.broadcast_to(jnp.asarray(w_), (nq,)).astype(q.dtype)
        M = jnp.diag(wv)
        e = q - params.q_default
        r = jnp.sqrt(e @ e + eps * eps)                  # softened norm: smooth + NaN-free at e=0
        speed2 = qd @ qd                                 # ‖q̇‖²  → HD2 (zero force at rest)
        a_des = -(speed2 * g_ * jnp.tanh(sh_ * r) / r) * e   # toward q_default; saturating; HD2
        f = -M @ a_des  # = wv · (speed2·gain·tanh(sharp·r)/r) · e   (per-joint weight cancels in a_des)
        return Spec(M, f)

    return leaf


def config_damping(nq: int, b: float = 2.0, mass: float = 1.0):
    """Pure joint-space damping: global dissipation + a full-rank metric contribution."""

    def leaf(q, qd, params):
        mass_, b_ = dynamic_gain(mass, params), dynamic_gain(b, params)
        M = mass_ * jnp.eye(nq, dtype=q.dtype)
        f = mass_ * b_ * qd  # = -M @ (-b q̇)  → isolated accel -b·q̇
        return Spec(M, f)

    return leaf


def speed_control(nq: int, b: float = 2.0, beta_speed=0.0, E_max=1e9, k_gate: float = 20.0,
                  mass: float = 1.0):
    """Config-space damping with a kinetic-energy cap — the speed-regulated successor to
    :func:`config_damping`. Two additive, metric-scaled terms:

    * **baseline damping ``b``** — the execution-speed knob. Isolated accel ``−b·q̇``; steady cruise
      ``≈ ‖a_des‖/b``, so a *lower* ``b`` runs *faster*. (This is what lets the rest of the damping
      stack come down once the cap below owns safety.)
    * **overspeed boost** — a HARD kinetic-energy ceiling for human-proximity safety: with
      ``E = ½‖q̇‖²``, the gate ``σ(k_gate·(E−E_max)) ∈ (0,1)`` is ~0 below the cap and →1 above it,
      adding strong damping ``beta_speed·M·q̇`` that bleeds energy until ``E ≤ E_max``. The smooth
      sigmoid (not a hard ``E>E_max`` step) keeps the command C1.

    NVlabs *decelerate-only* form: it never injects energy to reach a target speed (safe near a seated
    person). Config-space ``E`` bounds EVERY joint (the elbow toward a bystander too), not just the EE.
    ``beta_speed=0`` (the default) ⇒ ``b_eff=b`` ⇒ bit-identical to :func:`config_damping`.
    """

    def leaf(q, qd, params):
        b_, beta_, emax_, kg_, mass_ = (dynamic_gain(g, params)
                                        for g in (b, beta_speed, E_max, k_gate, mass))
        E = 0.5 * (qd @ qd)
        b_eff = b_ + beta_ * jax.nn.sigmoid(kg_ * (E - emax_))   # baseline + overspeed (KE-cap) damping
        M = mass_ * jnp.eye(nq, dtype=q.dtype)
        f = mass_ * b_eff * qd  # = -M @ (-b_eff q̇)  → isolated accel -b_eff·q̇
        return Spec(M, f)

    return leaf
