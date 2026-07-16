from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from datetime import datetime, timezone
import time

import yaml
import numpy as np
from ase import units
from ase.io import read, write
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.optimize import BFGS, FIRE


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
INPUT_MD_DIR = ROOT / "inputs" / "md"
OUTPUT_MD_DIR = ROOT / "outputs" / "md"

KNOWN_STRUCTURE_SUFFIXES = {
    ".traj",
    ".xyz",
    ".extxyz",
    ".xml",
    ".vasp",
    ".poscar",
    ".con",
    ".cif",
}

KNOWN_CONFIG_FILES = {
    "optimize": INPUT_MD_DIR / "optimize.yaml",
    "nvt": INPUT_MD_DIR / "nvt.yaml",
    "nve": INPUT_MD_DIR / "nve.yaml",
}

FINAL_STRUCTURE_FORMATS_WITH_VELOCITIES = {
    "traj",
    "extxyz",
}

DEFAULT_LOG_FIELDS = [
    "step",
    "time_fs",
    "temperature_K",
    "potential_energy_eV",
    "kinetic_energy_eV",
    "total_energy_eV",
]


class PrematureStopRequested(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def find_model_files() -> list[Path]:
    candidates = []
    for pattern in ("*.model", "*.pt", "*.pth"):
        candidates.extend(sorted(MODEL_DIR.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No model file found in {MODEL_DIR}. Put your fine-tuned MACE models there."
        )
    return candidates


def resolve_config_path(task: str, config_file: str | None = None) -> Path:
    if config_file is None:
        return KNOWN_CONFIG_FILES[task]

    candidate = Path(config_file).expanduser()
    if candidate.is_absolute():
        return candidate

    candidate_in_inputs = INPUT_MD_DIR / candidate
    if candidate_in_inputs.exists():
        return candidate_in_inputs

    candidate_relative_to_root = ROOT / candidate
    if candidate_relative_to_root.exists():
        return candidate_relative_to_root

    return candidate_relative_to_root


def parse_config_cli(task: str) -> Path:
    parser = argparse.ArgumentParser(
        description=f"Run MACE {task} using a YAML input file.",
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help=(
            "YAML input file. If omitted, the default file in inputs/md/ is used "
            f"({KNOWN_CONFIG_FILES[task].name}). Relative paths are resolved from "
            "inputs/md/ first, then from the project root."
        ),
    )
    args = parser.parse_args()
    return resolve_config_path(task, args.config)


def load_run_config(task: str, config_file: str | None = None) -> tuple[dict, Path]:
    config_path = resolve_config_path(task, config_file)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Fill this YAML file before launching {task}."
        )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}, config_path
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return payload, config_path


def select_model_files(config: dict) -> list[Path]:
    selected_models = config.get("selected_models")
    all_models = find_model_files()
    if selected_models is None:
        return all_models

    selected_set = set(selected_models)
    matched = [path for path in all_models if path.name in selected_set]
    missing = [name for name in selected_models if name not in {path.name for path in matched}]
    if missing:
        raise FileNotFoundError(f"Selected model files were not found in {MODEL_DIR}: {missing}")
    return matched


def model_output_dir(base_output_dir: Path, model_path: Path, structure_path: Path | None = None) -> Path:
    if structure_path is not None:
        path = base_output_dir / structure_path.stem / model_path.stem
    else:
        path = base_output_dir / model_path.stem
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_structure_path(structure_filename: str) -> Path:
    candidate = Path(structure_filename).expanduser()
    if candidate.is_absolute():
        if not candidate.exists():
            raise FileNotFoundError(f"Requested structure file does not exist: {candidate}")
        return candidate

    candidate_in_inputs = INPUT_MD_DIR / structure_filename
    if candidate_in_inputs.exists():
        return candidate_in_inputs

    candidate_relative_to_root = ROOT / structure_filename
    if candidate_relative_to_root.exists():
        return candidate_relative_to_root

    raise FileNotFoundError(
        f"Requested structure file was not found in {INPUT_MD_DIR} or relative to the project root: {structure_filename}"
    )


def find_structure_file(config: dict | None = None) -> Path:
    config = config or {}
    structure_filename = config.get("structure_file")
    if structure_filename:
        return _resolve_structure_path(str(structure_filename))

    preferred_names = [
        "POSCAR",
        "CONTCAR",
        "POSCAR-1",
        "CONTCAR-1",
        "structure.extxyz",
        "structure.traj",
        "structure.xyz",
        "structure.vasp",
        "vasprun.xml",
    ]
    for name in preferred_names:
        candidate = INPUT_MD_DIR / name
        if candidate.exists():
            return candidate

    candidates = sorted(
        path
        for path in INPUT_MD_DIR.iterdir()
        if path.is_file()
        and path.name != "README.md"
        and (
            path.suffix.lower() in KNOWN_STRUCTURE_SUFFIXES
            or path.suffix == ""
            or "POSCAR" in path.name
            or "CONTCAR" in path.name
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f"No structure file found in {INPUT_MD_DIR}. Put your POSCAR/CONTCAR/xyz/extxyz/traj/vasprun.xml there."
        )
    return candidates[0]


def select_structure_files(config: dict) -> list[Path]:
    structure_files = config.get("structure_files")
    structure_glob = config.get("structure_glob")

    if structure_files is not None and structure_glob is not None:
        raise ValueError("Use either `structure_files` or `structure_glob`, not both.")

    if structure_files is not None:
        if not isinstance(structure_files, list):
            raise ValueError("`structure_files` must be a YAML list of filenames.")
        return [_resolve_structure_path(str(name)) for name in structure_files]

    if structure_glob is not None:
        matches = sorted(INPUT_MD_DIR.glob(str(structure_glob)))
        matches = [path for path in matches if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"No structure files matched glob {structure_glob!r} in {INPUT_MD_DIR}")
        return matches

    return [find_structure_file(config)]


def load_calculator(model_path: Path, device: str = "cpu", default_dtype: str = "float32"):
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
    except Exception as exc:
        print(
            "Warning: cuequivariance could not be imported cleanly in this environment. "
            "Falling back to standard MACE execution without cuequivariance acceleration. "
            f"Reason: {exc.__class__.__name__}: {exc}"
        )
        sys.modules["cuequivariance"] = None
        sys.modules["cuequivariance_torch"] = None

    from mace.calculators import MACECalculator

    return MACECalculator(
        model_paths=str(model_path),
        device=device,
        default_dtype=default_dtype,
    )


def load_atoms(structure_path: Path):
    return read(str(structure_path), index=-1)


def prepare_output_dir(name: str) -> Path:
    path = OUTPUT_MD_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_named_output_dir(task: str, config: dict) -> Path:
    output_name = config.get("output_name", task)
    path = OUTPUT_MD_DIR / output_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_run_header(
    task: str,
    config_path: Path,
    config: dict,
    model_paths: list[Path],
    structure_paths: list[Path],
    output_root: Path,
) -> None:
    print("")
    print("=" * 80)
    print(f"Task: {task}")
    print(f"Config file: {config_path}")
    print(f"Output root: {output_root}")
    print(f"Device: {config.get('device', 'cpu')}")
    print(f"Default dtype: {config.get('default_dtype', 'float32')}")
    print(f"Models selected ({len(model_paths)}):")
    for model_path in model_paths:
        print(f"  - {model_path}")
    print(f"Structures selected ({len(structure_paths)}):")
    for structure_path in structure_paths:
        print(f"  - {structure_path}")
    print("=" * 80)
    print("")


def print_structure_header(task: str, structure_path: Path, structure_index: int, total_structures: int) -> None:
    print("")
    print("-" * 80)
    print(f"{task.upper()} structure {structure_index}/{total_structures}")
    print(f"Geometry input: {structure_path}")
    print("-" * 80)


def print_model_header(
    task: str,
    model_path: Path,
    model_index: int,
    total_models: int,
    output_dir: Path,
) -> None:
    print(f"Calculation: {task.upper()}")
    print(f"Model {model_index}/{total_models}: {model_path}")
    print(f"Output directory: {output_dir}")


def get_output_file_settings(config: dict, key_prefix: str, default_name: str, default_format: str) -> tuple[str, str]:
    filename = config.get(f"{key_prefix}_file", default_name)
    file_format = config.get(f"{key_prefix}_format", default_format)
    return str(filename), str(file_format)


def _write_lammps_dump_frame(path: Path, atoms, step: int, append: bool = True) -> None:
    cell = atoms.cell
    lengths = cell.lengths()
    angles = cell.angles()
    if not np.allclose(angles, [90.0, 90.0, 90.0], atol=1e-8):
        raise ValueError("Custom LAMMPS dump writer currently supports orthorhombic cells only.")

    positions = atoms.get_positions()
    velocities = atoms.get_velocities()
    if velocities is None:
        velocities = np.zeros_like(positions)
    forces = atoms.get_forces() if atoms.calc is not None else np.zeros_like(positions)

    species = sorted(set(atoms.get_chemical_symbols()))
    type_map = {symbol: index + 1 for index, symbol in enumerate(species)}
    mode = "a" if append else "w"

    with path.open(mode, encoding="utf-8") as handle:
        handle.write("ITEM: TIMESTEP\n")
        handle.write(f"{step}\n")
        handle.write("ITEM: NUMBER OF ATOMS\n")
        handle.write(f"{len(atoms)}\n")
        handle.write("ITEM: BOX BOUNDS pp pp pp\n")
        handle.write(f"0.0 {lengths[0]:.16f}\n")
        handle.write(f"0.0 {lengths[1]:.16f}\n")
        handle.write(f"0.0 {lengths[2]:.16f}\n")
        handle.write("ITEM: ATOMS id type element x y z vx vy vz fx fy fz\n")
        for atom_index, (symbol, pos, vel, frc) in enumerate(
            zip(atoms.get_chemical_symbols(), positions, velocities, forces),
            start=1,
        ):
            handle.write(
                f"{atom_index} {type_map[symbol]} {symbol} "
                f"{pos[0]:.16f} {pos[1]:.16f} {pos[2]:.16f} "
                f"{vel[0]:.16f} {vel[1]:.16f} {vel[2]:.16f} "
                f"{frc[0]:.16f} {frc[1]:.16f} {frc[2]:.16f}\n"
            )


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _normalize_yaml_indices(raw_indices, natoms: int, index_base: int = 1) -> list[int]:
    if not isinstance(raw_indices, list) or not raw_indices:
        raise ValueError("Distance-monitor atom indices must be provided as a non-empty YAML list.")
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


def _group_com_z(atoms, indices: list[int]) -> float:
    subset = atoms[indices]
    return float(subset.get_center_of_mass()[2])


def _group_average_z(atoms, indices: list[int]) -> float:
    return float(atoms.positions[indices, 2].mean())


def _build_distance_monitors(config: dict, atoms) -> list[dict]:
    entries = config.get("monitored_distances", [])
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise ValueError("`monitored_distances` must be a YAML list.")

    natoms = len(atoms)
    monitors = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each `monitored_distances` entry must be a YAML mapping.")
        name = str(entry.get("name", "")).strip()
        kind = str(entry.get("kind", "atom_distance")).strip().lower()
        if not name:
            raise ValueError("Each monitored distance must define a non-empty `name`.")
        index_base = int(entry.get("index_base", 1))

        if kind == "atom_distance":
            indices = _normalize_yaml_indices(entry.get("atom_indices"), natoms, index_base=index_base)
            if len(indices) != 2:
                raise ValueError("`atom_distance` monitors require exactly two atom indices.")
            monitors.append({"name": name, "kind": kind, "indices": indices})
            continue

        if kind in {"com_z_distance", "average_z_distance"}:
            group_a = _normalize_yaml_indices(entry.get("group_a_indices"), natoms, index_base=index_base)
            group_b = _normalize_yaml_indices(entry.get("group_b_indices"), natoms, index_base=index_base)
            monitors.append(
                {
                    "name": name,
                    "kind": kind,
                    "group_a": group_a,
                    "group_b": group_b,
                }
            )
            continue

        raise ValueError(
            "Unsupported distance-monitor kind. Use one of: "
            "atom_distance, com_z_distance, average_z_distance"
        )
    return monitors


def _evaluate_distance_monitor(atoms, monitor: dict) -> float:
    kind = monitor["kind"]
    if kind == "atom_distance":
        i, j = monitor["indices"]
        return float(atoms.get_distance(i, j, mic=True))
    if kind == "com_z_distance":
        return abs(_group_com_z(atoms, monitor["group_a"]) - _group_com_z(atoms, monitor["group_b"]))
    if kind == "average_z_distance":
        return abs(_group_average_z(atoms, monitor["group_a"]) - _group_average_z(atoms, monitor["group_b"]))
    raise ValueError(f"Unsupported distance-monitor kind: {kind}")


def _get_log_fields(config: dict) -> list[str]:
    log_fields = config.get("log_fields", DEFAULT_LOG_FIELDS)
    if not isinstance(log_fields, list) or not log_fields:
        raise ValueError("`log_fields` must be a non-empty YAML list.")
    return [str(field) for field in log_fields]


def _collect_log_values(dynamics, atoms, config: dict, distance_monitors: list[dict]) -> dict[str, float]:
    step = int(getattr(dynamics, "nsteps", 0))
    time_step_fs = float(config.get("time_step_fs", 0.0))
    values = {
        "step": step,
        "time_fs": step * time_step_fs if time_step_fs > 0.0 else float("nan"),
        "temperature_K": _safe_float(atoms.get_temperature()),
        "potential_energy_eV": _safe_float(atoms.get_potential_energy()),
        "kinetic_energy_eV": _safe_float(atoms.get_kinetic_energy()),
        "total_energy_eV": float("nan"),
        "natoms": len(atoms),
    }
    if np.isfinite(values["potential_energy_eV"]) and np.isfinite(values["kinetic_energy_eV"]):
        values["total_energy_eV"] = values["potential_energy_eV"] + values["kinetic_energy_eV"]

    forces = None
    try:
        forces = atoms.get_forces()
    except Exception:
        forces = None
    values["max_force_eV_A"] = (
        float(np.linalg.norm(forces, axis=1).max()) if forces is not None and len(forces) else float("nan")
    )

    for monitor in distance_monitors:
        values[monitor["name"]] = _evaluate_distance_monitor(atoms, monitor)
    return values


def _format_log_value(field: str, value) -> str:
    if field in {"step", "natoms"}:
        return f"{int(value):>12d}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):>12d}"
    if isinstance(value, float):
        if np.isnan(value):
            return f"{'nan':>16s}"
        return f"{value:>16.8f}"
    return f"{str(value):>16s}"


def _write_log_header(log_path: Path, task: str, fields: list[str], distance_monitors: list[dict], append: bool) -> None:
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as handle:
        handle.write(f"# task: {task}\n")
        if distance_monitors:
            handle.write("# monitored_distances:\n")
            for monitor in distance_monitors:
                handle.write(f"#   - {monitor['name']} ({monitor['kind']})\n")
        handle.write(" ".join(f"{field:>16s}" for field in fields) + "\n")


def attach_runtime_logger(dynamics, atoms, output_dir: Path, config: dict) -> None:
    task = str(config.get("_task_name", "md"))
    log_file = str(config.get("log_file", "run.log"))
    log_interval = int(config.get("log_interval", 10))
    append_log = bool(config.get("append_log", False))
    if log_interval < 1:
        raise ValueError("`log_interval` must be >= 1.")

    log_path = output_dir / log_file
    log_fields = _get_log_fields(config)
    distance_monitors = _build_distance_monitors(config, atoms)
    for monitor in distance_monitors:
        if monitor["name"] not in log_fields:
            log_fields.append(monitor["name"])

    _write_log_header(log_path, task, log_fields, distance_monitors, append=append_log)

    def write_log_line() -> None:
        values = _collect_log_values(dynamics, atoms, config, distance_monitors)
        line = " ".join(_format_log_value(field, values.get(field, float("nan"))) for field in log_fields)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    dynamics.attach(write_log_line, interval=log_interval)


def save_final_structure_enabled(config: dict) -> bool:
    return bool(config.get("save_final_structure", True))


def save_final_structure(atoms, output_dir: Path, config: dict) -> None:
    if not save_final_structure_enabled(config):
        print("Final structure writing disabled by input flag `save_final_structure: false`.")
        return

    filename, file_format = get_output_file_settings(
        config,
        key_prefix="final_structure",
        default_name="final_structure.extxyz",
        default_format="extxyz",
    )
    normalized_format = file_format.lower()
    velocities = atoms.get_velocities()
    if velocities is not None and normalized_format not in FINAL_STRUCTURE_FORMATS_WITH_VELOCITIES:
        print(
            "Warning: the selected final structure format does not reliably preserve velocities. "
            f"Use one of {sorted(FINAL_STRUCTURE_FORMATS_WITH_VELOCITIES)} for restartable outputs."
        )
    write(str(output_dir / filename), atoms, format=file_format)


def attach_trajectory_and_log(dynamics, atoms, output_dir: Path, config: dict) -> None:
    trajectory_file, trajectory_format = get_output_file_settings(
        config,
        key_prefix="trajectory",
        default_name="trajectory.extxyz",
        default_format="extxyz",
    )
    trajectory_interval = int(config.get("trajectory_interval", config.get("log_interval", 10)))
    append_trajectory = bool(config.get("append_trajectory", True))

    trajectory_path = output_dir / trajectory_file

    def write_frame() -> None:
        normalized_format = trajectory_format.lower()
        if normalized_format in {"lammps-dump-text", "lammpstrj", "lammps-dump"}:
            _write_lammps_dump_frame(
                trajectory_path,
                atoms,
                step=getattr(dynamics, "nsteps", 0),
                append=append_trajectory,
            )
            return
        write(str(trajectory_path), atoms, format=trajectory_format, append=append_trajectory)

    dynamics.attach(write_frame, interval=trajectory_interval)
    attach_runtime_logger(dynamics, atoms, output_dir, config)


def get_stopcar_path(output_dir: Path, config: dict) -> Path:
    return output_dir / str(config.get("stopcar_file", "STOPCAR"))


def write_stopcar(path: Path, reason: str) -> None:
    path.write_text(f"{reason.rstrip()}\n", encoding="utf-8")


def load_custom_stop_condition(config: dict):
    if not bool(config.get("stop_on_custom_condition", False)):
        return None

    module_name = config.get("stop_condition_module")
    function_name = config.get("stop_condition_function")
    if not module_name or not function_name:
        raise ValueError(
            "`stop_on_custom_condition: true` requires both "
            "`stop_condition_module` and `stop_condition_function`."
        )

    module = importlib.import_module(str(module_name))
    condition = getattr(module, str(function_name))
    if not callable(condition):
        raise TypeError(f"Configured stop condition is not callable: {module_name}.{function_name}")
    return condition


def build_stop_monitor(dynamics, atoms, output_dir: Path, config: dict):
    enable_stopcar = bool(config.get("enable_stopcar", True))
    stopcar_path = get_stopcar_path(output_dir, config)
    custom_condition = load_custom_stop_condition(config)
    create_stopcar_on_custom_condition = bool(config.get("create_stopcar_on_custom_condition", True))

    def check_stop() -> None:
        if enable_stopcar and stopcar_path.exists():
            raise PrematureStopRequested(f"STOPCAR detected at {stopcar_path}")

        if custom_condition is None:
            return

        result = custom_condition(
            atoms=atoms,
            dynamics=dynamics,
            config=config,
            output_dir=output_dir,
        )

        if isinstance(result, tuple):
            should_stop, reason = result
        else:
            should_stop, reason = bool(result), None

        if not should_stop:
            return

        stop_reason = reason or "Custom stop condition triggered."
        if create_stopcar_on_custom_condition:
            write_stopcar(stopcar_path, stop_reason)
        raise PrematureStopRequested(stop_reason)

    return check_stop


def attach_stop_monitor(dynamics, atoms, output_dir: Path, config: dict) -> Path:
    stop_check_interval = int(config.get("stop_check_interval", 1))
    if stop_check_interval < 1:
        raise ValueError("`stop_check_interval` must be >= 1.")
    stopcar_path = get_stopcar_path(output_dir, config)
    dynamics.attach(build_stop_monitor(dynamics, atoms, output_dir, config), interval=stop_check_interval)
    return stopcar_path


def build_optimizer(name: str, atoms, output_dir: Path, config: dict):
    logfile = output_dir / str(config.get("optimizer_log_file", "optimizer.log"))
    restart = output_dir / str(config.get("optimizer_restart_file", "optimizer.restart"))
    if name.lower() == "fire":
        return FIRE(atoms, logfile=str(logfile), restart=str(restart))
    if name.lower() == "bfgs":
        return BFGS(atoms, logfile=str(logfile), restart=str(restart))
    raise ValueError("optimizer must be 'fire' or 'bfgs'")


def initialize_velocities(
    atoms,
    temperature_k: float,
    remove_translation: bool = True,
    remove_rotation: bool = True,
) -> None:
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_k)
    if remove_translation:
        Stationary(atoms)
    if remove_rotation:
        ZeroRotation(atoms)


def setup_nvt(atoms, config: dict):
    thermostat = str(config.get("thermostat", "nose-hoover")).lower()
    temperature_k = float(config.get("temperature_k", 300.0))
    time_step_fs = float(config.get("time_step_fs", 1.0))
    initialize = bool(config.get("initialize_velocities", True))
    remove_translation = bool(config.get("remove_translation", True))
    remove_rotation = bool(config.get("remove_rotation", True))

    if initialize:
        initialize_velocities(
            atoms,
            temperature_k=temperature_k,
            remove_translation=remove_translation,
            remove_rotation=remove_rotation,
        )

    if thermostat == "nose-hoover":
        tdamp_fs = float(config.get("tdamp_fs", 100.0))
        tchain = int(config.get("tchain", 3))
        tloop = int(config.get("tloop", 1))
        return NoseHooverChainNVT(
            atoms,
            timestep=time_step_fs * units.fs,
            temperature_K=temperature_k,
            tdamp=tdamp_fs * units.fs,
            tchain=tchain,
            tloop=tloop,
        )

    if thermostat == "langevin":
        friction = float(config.get("friction", 0.02))
        return Langevin(
            atoms,
            timestep=time_step_fs * units.fs,
            temperature_K=temperature_k,
            friction=friction,
        )

    if thermostat == "berendsen":
        taut_fs = float(config.get("taut_fs", 100.0))
        return NVTBerendsen(
            atoms,
            timestep=time_step_fs * units.fs,
            temperature_K=temperature_k,
            taut=taut_fs * units.fs,
        )

    if thermostat == "bussi":
        taut_fs = float(config.get("taut_fs", 100.0))
        return Bussi(
            atoms,
            timestep=time_step_fs * units.fs,
            temperature_K=temperature_k,
            taut=taut_fs * units.fs,
        )

    raise ValueError(
        "Unsupported thermostat. Use one of: nose-hoover, langevin, berendsen, bussi"
    )


def setup_nve(atoms, time_step_fs: float):
    if "momenta" not in atoms.arrays:
        raise ValueError(
            "NVE needs velocities already present in the input file. Use a trajectory frame or an extxyz with velocities."
        )
    return VelocityVerlet(atoms, timestep=time_step_fs * units.fs)


def get_timing_file(config: dict) -> str:
    return str(config.get("timing_file", "timing_summary.yaml"))


def measure_run(run_callable):
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    stop_reason = None
    stopped_early = False
    try:
        run_callable()
    except PrematureStopRequested as exc:
        stop_reason = exc.reason
        stopped_early = True
    elapsed_seconds = time.perf_counter() - start
    finished_at = datetime.now(timezone.utc).isoformat()
    return elapsed_seconds, started_at, finished_at, stopped_early, stop_reason


def build_timing_summary(
    task: str,
    config: dict,
    requested_steps: int,
    completed_steps: int | None,
    elapsed_seconds: float,
    started_at: str,
    finished_at: str,
    stopped_early: bool = False,
    stop_reason: str | None = None,
) -> dict:
    payload = {
        "task": task,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "elapsed_wall_time_s": float(elapsed_seconds),
        "elapsed_wall_time_min": float(elapsed_seconds / 60.0),
        "requested_steps": int(requested_steps),
        "completed_steps": None if completed_steps is None else int(completed_steps),
        "average_wall_time_per_step_s": None,
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
    }

    if completed_steps and completed_steps > 0:
        payload["average_wall_time_per_step_s"] = float(elapsed_seconds / completed_steps)

    if task in {"nvt", "nve"}:
        time_step_fs = float(config.get("time_step_fs", 1.0))
        completed = int(completed_steps or 0)
        simulated_time_fs = completed * time_step_fs
        payload["time_step_fs"] = time_step_fs
        payload["simulated_time_fs"] = float(simulated_time_fs)
        payload["simulated_time_ps"] = float(simulated_time_fs / 1000.0)

    return payload


def write_timing_summary(output_dir: Path, config: dict, summary: dict) -> None:
    (output_dir / get_timing_file(config)).write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )


def print_timing_summary(summary: dict) -> None:
    completed_steps = summary.get("completed_steps")
    average = summary.get("average_wall_time_per_step_s")
    print(f"Wall time: {summary['elapsed_wall_time_s']:.3f} s")
    if completed_steps is not None:
        print(f"Completed steps: {completed_steps}")
    if average is not None:
        print(f"Average wall time per step: {average:.6f} s")
    if summary.get("stopped_early"):
        print(f"Stopped early: True")
        print(f"Stop reason: {summary.get('stop_reason')}")


def write_run_settings(
    output_dir: Path,
    config: dict,
    model_path: Path,
    structure_path: Path,
    config_path: Path | None = None,
    timing_summary: dict | None = None,
) -> None:
    payload = {
        "model_file": model_path.name,
        "structure_file": structure_path.name,
        "settings": config,
    }
    if config_path is not None:
        payload["config_file"] = str(config_path)
    if timing_summary is not None:
        payload["timing"] = timing_summary
    (output_dir / "run_settings.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
