"""Finsler / Lagrangian execution energies and their second-order specs.

An *execution energy* ``L_e(x, xd)`` — homogeneous of degree 2 in ``xd`` — gives a fabric its
notion of "speed" to conserve. From ``L_e`` alone, autodiff produces the energy spec
``(M_e, f_e)``:

    M_e = d2 L_e / dxd2                         (the energy/mass metric; SPD for a valid energy)
    f_e = d/dx(dL_e/dxd) . xd  -  dL_e/dx       (the Coriolis force, evaluated at xddot = 0)

so the energy's own dynamics is ``M_e xddot + f_e = 0`` and — the identity the whole framework
hinges on — the rate of energy change along any trajectory is

    dH_e/dt = xd^T (M_e xddot + f_e).

The energization operator (:mod:`fabrix.geometry`) zeroes exactly this quantity. As with the
task Jacobian and curvature term, the mass metric and Coriolis force are *autodiff*, not
hand-derived: a Lagrangian closure in, a ``(M_e, f_e)`` callable out.

Convention: an ``Energy`` is a pure callable ``energy(x, xd) -> (M_e, f_e)``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def energy_spec(L_e, x, xd):
    """Build the energy spec ``(M_e, f_e)`` from a Lagrangian ``L_e(x, xd) -> scalar``.

    ``M_e = d2L/dxd2`` (Hessian in velocity) and ``f_e = d/dx(dL/dxd).xd - dL/dx`` are the
    Euler-Lagrange operator split into its ``xddot`` coefficient and its velocity-only part.
    All three derivatives come from autodiff, so any HD2 ``L_e`` works without hand-derivation.
    """
    dL_dxd = jax.grad(L_e, argnums=1)
    M = jax.jacfwd(dL_dxd, argnums=1)(x, xd)                      # d2L/dxd2
    # d/dt (dL/dxd) with xddot = 0  ==  d/dx(dL/dxd) . xd, a jvp in x along xd
    dpx = jax.jvp(lambda xx: dL_dxd(xx, xd), (x,), (xd,))[1]
    f = dpx - jax.grad(L_e, argnums=0)(x, xd)
    return M, f


def fixed_metric_energy(n: int, dtype=jnp.float32):
    """The simplest valid execution energy: ``L_e = 1/2 ||xd||^2`` -> ``M_e = I``, ``f_e = 0``.

    Conserving this energy is conserving ``1/2 ||xd||^2`` (kinetic energy at unit mass), so an
    energized geometry traces its path at constant speed. Analytic (no autodiff) since the
    metric is constant — cheaper, and the natural default for a position-space fabric.
    """
    eye = jnp.eye(n, dtype=dtype)
    zero = jnp.zeros(n, dtype=dtype)

    def energy(x, xd):
        return eye, zero

    return energy


def lagrangian_energy(G_fn):
    """Energy from a configuration-dependent metric: ``L_e = 1/2 xd^T G(x) xd`` (HD2 in xd).

    ``G_fn(x) -> (n, n)`` must be SPD. Then ``M_e = G(x)`` and ``f_e`` is the induced Coriolis
    force (the Christoffel terms of ``G``), both via :func:`energy_spec`. Use this to shape how
    the fabric trades speed across directions; :func:`fixed_metric_energy` is the ``G = I`` case.
    """

    def L_e(x, xd):
        return 0.5 * xd @ (G_fn(x) @ xd)

    def energy(x, xd):
        return energy_spec(L_e, x, xd)

    return energy
