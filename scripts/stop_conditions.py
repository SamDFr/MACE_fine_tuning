from __future__ import annotations

import math


def _get_required_config(config: dict, key: str):
    if key not in config:
        raise ValueError(f"Missing required stop-condition setting: {key}")
    return config[key]


def _normalize_indices(raw_indices, natoms: int, index_base: int) -> list[int]:
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("Stop-condition atom indices must be provided as a non-empty YAML list.")

    normalized = []
    for value in raw_indices:
        index = int(value) - index_base
        if index < 0 or index >= natoms:
            raise IndexError(
                f"Atom index {value} is out of range for a system with {natoms} atoms "
                f"when using index base {index_base}."
            )
        normalized.append(index)
    return normalized


def _get_state(dynamics, key: str) -> dict:
    if not hasattr(dynamics, "_mace_stop_states"):
        dynamics._mace_stop_states = {}
    if key not in dynamics._mace_stop_states:
        dynamics._mace_stop_states[key] = {}
    return dynamics._mace_stop_states[key]


def _compute_group_com_z(atoms, indices: list[int]) -> float:
    subset = atoms[indices]
    return float(subset.get_center_of_mass()[2])


def _compute_group_average_z(atoms, indices: list[int]) -> float:
    return float(atoms.positions[indices, 2].mean())


def gas_surface_scattering_reflection_stop(*, atoms, dynamics, config: dict, output_dir) -> tuple[bool, str | None]:
    """
    Reflection-aware stop condition for gas-surface scattering trajectories.

    Required YAML keys:
      scattering_molecule_indices: [1, 2]
      scattering_surface_top_indices: [259, 260, ...]
      scattering_activation_distance_a: 6.0
      scattering_stop_distance_a: 8.0

    Optional YAML keys:
      scattering_index_base: 1
      scattering_surface_z_mode: average
      scattering_require_outgoing: true
      scattering_distance_hysteresis_a: 1.0e-3

    Logic:
      1. The molecule/surface distance is monitored along z.
      2. The stop criterion is "armed" only after the molecule has first come
         closer than `scattering_activation_distance_a`.
      3. Once armed, the run stops when the molecule moves away and the distance
         exceeds `scattering_stop_distance_a`.
    """

    natoms = len(atoms)
    index_base = int(config.get("scattering_index_base", 1))
    molecule_indices = _normalize_indices(
        _get_required_config(config, "scattering_molecule_indices"),
        natoms,
        index_base,
    )
    surface_indices = _normalize_indices(
        _get_required_config(config, "scattering_surface_top_indices"),
        natoms,
        index_base,
    )

    activation_distance = float(_get_required_config(config, "scattering_activation_distance_a"))
    stop_distance = float(_get_required_config(config, "scattering_stop_distance_a"))
    require_outgoing = bool(config.get("scattering_require_outgoing", True))
    hysteresis = float(config.get("scattering_distance_hysteresis_a", 1.0e-3))
    surface_z_mode = str(config.get("scattering_surface_z_mode", "average")).lower()

    if stop_distance <= activation_distance:
        raise ValueError(
            "`scattering_stop_distance_a` must be larger than "
            "`scattering_activation_distance_a` for a reflection-aware stop."
        )

    molecule_com_z = _compute_group_com_z(atoms, molecule_indices)
    if surface_z_mode == "average":
        surface_z = _compute_group_average_z(atoms, surface_indices)
    elif surface_z_mode == "com":
        surface_z = _compute_group_com_z(atoms, surface_indices)
    else:
        raise ValueError("`scattering_surface_z_mode` must be either 'average' or 'com'.")

    distance = abs(molecule_com_z - surface_z)

    state = _get_state(dynamics, "gas_surface_scattering_reflection_stop")
    previous_distance = state.get("previous_distance")
    minimum_distance = min(distance, state.get("minimum_distance", math.inf))
    armed = bool(state.get("armed", False))

    if minimum_distance <= activation_distance:
        armed = True

    moving_away = False
    if previous_distance is not None:
        moving_away = distance > previous_distance + hysteresis

    state["previous_distance"] = distance
    state["minimum_distance"] = minimum_distance
    state["armed"] = armed
    state["last_molecule_com_z"] = molecule_com_z
    state["last_surface_z"] = surface_z
    state["last_distance"] = distance
    state["output_dir"] = str(output_dir)

    if not armed:
        return False, None

    if distance < stop_distance:
        return False, None

    if require_outgoing and not moving_away:
        return False, None

    reason = (
        "Gas-surface scattering stop condition triggered: "
        f"molecule-surface distance = {distance:.6f} A, "
        f"minimum visited distance = {minimum_distance:.6f} A, "
        f"activation distance = {activation_distance:.6f} A, "
        f"stop distance = {stop_distance:.6f} A."
    )
    return True, reason
