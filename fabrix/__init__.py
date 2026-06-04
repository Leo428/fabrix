"""fabrix — a JAX-native geometric fabrics motion-generation library.

Milestone 1: forced attractor fabric. Public surface grows as milestones land.
"""

from fabrix.diff import value_jac_curv
from fabrix.fabric import Fabric, FabricParams
from fabrix.integrate import rollout, step
from fabrix.kinematics import CustomFK, KinematicsProvider, MJXProvider
from fabrix.leaves import attractor, config_damping, posture
from fabrix.maps import site_position_map
from fabrix.spec import Spec, combine, pullback, resolve

__all__ = [
    "Spec", "pullback", "combine", "resolve",
    "value_jac_curv",
    "KinematicsProvider", "CustomFK", "MJXProvider",
    "site_position_map",
    "attractor", "posture", "config_damping",
    "Fabric", "FabricParams",
    "step", "rollout",
]
