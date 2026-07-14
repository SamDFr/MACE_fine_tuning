from __future__ import annotations

from pathlib import Path

import yaml
import numpy as np
from ase import units
from ase.io import read, write
from ase.md import MDLogger
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


def find_model_files() -> list[Path]:
    candidates = []
    for pattern in ("*.model", "*.pt", "*.pth"):
        candidates.extend(sorted(MODEL_DIR.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(
            f"No model file found in {MODEL_DIR}. Put your fine-tuned MACE models there."
        )
    return candidates


def load_run_config(task: str) -> dict:
    config_path = KNOWN_CONFIG_FILES[task]
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config file: {config_path}. Fill this YAML file before launching {task}."
        )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return payload


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


def save_final_structure(atoms, output_dir: Path, config: dict) -> None:
    filename, file_format = get_output_file_settings(
        config,
        key_prefix="final_structure",
        default_name="final_structure.extxyz",
        default_format="extxyz",
    )
    write(str(output_dir / filename), atoms, format=file_format)


def attach_trajectory_and_log(dynamics, atoms, output_dir: Path, config: dict) -> None:
    trajectory_file, trajectory_format = get_output_file_settings(
        config,
        key_prefix="trajectory",
        default_name="trajectory.extxyz",
        default_format="extxyz",
    )
    log_file = str(config.get("log_file", "run.log"))
    trajectory_interval = int(config.get("trajectory_interval", config.get("log_interval", 10)))
    log_interval = int(config.get("log_interval", 10))
    append_trajectory = bool(config.get("append_trajectory", True))

    trajectory_path = output_dir / trajectory_file
    log_path = output_dir / log_file

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
    logger = MDLogger(dynamics, atoms, str(log_path), header=True, stress=False, peratom=False)
    dynamics.attach(logger, interval=log_interval)


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


def write_run_settings(output_dir: Path, config: dict, model_path: Path, structure_path: Path) -> None:
    payload = {
        "model_file": model_path.name,
        "structure_file": structure_path.name,
        "settings": config,
    }
    (output_dir / "run_settings.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
