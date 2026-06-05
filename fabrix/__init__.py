"""fabrix — a JAX-native geometric fabrics motion-generation library.

Milestone 1: forced attractor fabric. Public surface grows as milestones land.
"""

from fabrix.collision import (
    SphereModel, arm_obstacle_geometry, arm_obstacle_potential, arm_plane_geometry,
    arm_plane_potential, auto_arm_spheres, load_spheres, nonadjacent_pairs,
    self_collision_geometry, self_collision_potential,
)
from fabrix.diff import value_jac_curv
from fabrix.energy import energy_spec, fixed_metric_energy, lagrangian_energy
from fabrix.fabric import Fabric, FabricParams, GeometricFabric
from fabrix.geometry import (
    energize, joint_limit_geometry, joint_limit_potential, obstacle_geometry,
    obstacle_potential, plane_geometry, plane_potential, sdf_barrier_geometry,
    sdf_barrier_potential,
)
from fabrix.integrate import rollout, step
from fabrix.kinematics import CustomFK, KinematicsProvider
from fabrix.leaves import attractor, config_damping, pose_attractor, posture
from fabrix.maps import (
    plane_sdf_map, se3_pose_error_map, site_position_map, sphere_sdf_map,
)
from fabrix.spec import Spec, combine, dynamic_gain, pullback, resolve

__all__ = [
    "Spec", "pullback", "combine", "resolve", "dynamic_gain",
    "value_jac_curv",
    "KinematicsProvider", "CustomFK",
    "site_position_map", "sphere_sdf_map", "plane_sdf_map", "se3_pose_error_map",
    "attractor", "pose_attractor", "posture", "config_damping",
    "energy_spec", "fixed_metric_energy", "lagrangian_energy",
    "energize", "joint_limit_geometry", "joint_limit_potential",
    "obstacle_geometry", "obstacle_potential", "plane_geometry", "plane_potential",
    "sdf_barrier_geometry", "sdf_barrier_potential",
    "SphereModel", "auto_arm_spheres", "nonadjacent_pairs", "load_spheres",
    "self_collision_geometry", "self_collision_potential",
    "arm_obstacle_geometry", "arm_obstacle_potential",
    "arm_plane_geometry", "arm_plane_potential",
    "Fabric", "FabricParams", "GeometricFabric",
    "step", "rollout",
]
