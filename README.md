# MACE Fine-Tuning Analysis and Molecular Dynamics

This repository provides a lightweight workflow for:

- evaluating one or more fine-tuned MACE models against reference data
- generating correlation plots for energies and atomic forces
- running structure optimization, NVT molecular dynamics, and NVE molecular dynamics

The project is organized around simple input, model, and output directories rather than a packaged application framework.

## Repository Layout

```text
.
├── model/
├── inputs/
│   ├── correlation/
│   └── md/
├── outputs/
│   ├── correlation/
│   └── md/
├── notebooks/
│   ├── 01_correlation_plots.ipynb
│   └── 02_check_md_inputs.ipynb
└── scripts/
    ├── optimize_structure.py
    ├── run_nvt.py
    └── run_nve.py
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To work with notebooks:

```bash
source .venv/bin/activate
jupyter notebook
```

## Models

Place fine-tuned MACE model files in `model/`.

Supported model extensions:

- `.model`
- `.pt`
- `.pth`

Example:

```text
model/
├── seed_1.model
├── seed_2.model
└── seed_3.model
```

Model selection for MD and optimization is controlled through YAML:

- `selected_models: null` uses all detected models in `model/`
- `selected_models: ['seed_2.model']` restricts the run to specific model files

## Correlation Analysis Inputs

Place reference files in `inputs/correlation/`.

Supported formats:

- `vasprun*.xml`
- `*.traj`
- `*.extxyz`
- `*.xyz`

For `vasprun.xml` and ASE trajectory files, the notebook reads reference energies and forces from the attached calculator when available.

## MD Inputs

Place structure files and YAML input files in `inputs/md/`.

Supported structure formats include:

- `POSCAR`
- `CONTCAR`
- `.traj`
- `.xyz`
- `.extxyz`
- `.vasp`
- `.cif`
- `vasprun.xml`

The MD scripts can process:

- a single structure file
- an explicit list of structure files
- a glob pattern

Examples:

```yaml
structure_file: POSCAR-1
```

```yaml
structure_files:
  - POSCAR-160
  - POSCAR-161
  - POSCAR-162
```

```yaml
structure_glob: "POSCAR-*"
```

Subdirectories under `inputs/md/` are supported when explicitly referenced, for example:

```yaml
structure_file: 100traj_AIMD/batch_1/POSCAR-1
```

or:

```yaml
structure_glob: "100traj_AIMD/batch_1/POSCAR-*"
```

## Notebooks

### Correlation Notebook

Use [notebooks/01_correlation_plots.ipynb](/Users/samuel/Desktop/postdoc_PhLAM/codes/MACE_fine_tuning/notebooks/01_correlation_plots.ipynb).

The notebook:

- detects all models in `model/`
- detects reference files in `inputs/correlation/`
- supports multiple models and multiple input files
- writes results to `outputs/correlation/`

Generated outputs include:

- one energy parity plot per model
- one force norm parity plot per model
- `Fx`, `Fy`, `Fz` parity plots per species and per model
- force-direction analysis per model
- a global model-comparison summary

Useful notebook settings:

- `SELECTED_FILES = None` to process all detected files
- `SELECTED_FILES = ['file1.xml', 'file2.traj']` to restrict the analysis
- `FRAME_STRIDE = 1` to use all frames
- `FRAME_STRIDE = 10` to keep one frame out of ten

### MD Input Check Notebook

Use [notebooks/02_check_md_inputs.ipynb](/Users/samuel/Desktop/postdoc_PhLAM/codes/MACE_fine_tuning/notebooks/02_check_md_inputs.ipynb).

The notebook is intended for local inspection before running MD. It helps verify:

- detected models
- detected structure files
- whether velocities are available for NVE
- the structure read by ASE

## Command-Line Usage

Each MD script accepts an optional YAML file path as a positional argument.

If no argument is provided, the default file is used:

- `inputs/md/optimize.yaml`
- `inputs/md/nvt.yaml`
- `inputs/md/nve.yaml`

Relative YAML paths are resolved from `inputs/md/` first, then from the project root.

### Structure Optimization

```bash
python3 scripts/optimize_structure.py
python3 scripts/optimize_structure.py inputs/md/optimize.yaml
python3 scripts/optimize_structure.py inputs/md/my_optimize_batch.yaml
```

Key YAML options:

- `structure_file`
- `structure_files`
- `structure_glob`
- `selected_models`
- `device`
- `default_dtype`
- `output_name`
- `trajectory_file`
- `trajectory_format`
- `final_structure_file`
- `final_structure_format`
- `save_final_structure`
- `log_file`
- `append_log`
- `log_fields`
- `timing_file`
- `append_trajectory`
- `trajectory_interval`
- `optimizer_log_file`
- `optimizer_restart_file`
- `enable_stopcar`
- `stopcar_file`
- `stop_check_interval`
- `stop_on_custom_condition`
- `stop_condition_module`
- `stop_condition_function`
- `create_stopcar_on_custom_condition`
- `optimizer`
- `fmax`
- `steps`
- `log_interval`

### NVT Molecular Dynamics

```bash
python3 scripts/run_nvt.py
python3 scripts/run_nvt.py inputs/md/nvt.yaml
python3 scripts/run_nvt.py inputs/md/nvt_batch_300K.yaml
```

Key YAML options:

- `structure_file`
- `structure_files`
- `structure_glob`
- `selected_models`
- `device`
- `default_dtype`
- `output_name`
- `trajectory_file`
- `trajectory_format`
- `final_structure_file`
- `final_structure_format`
- `save_final_structure`
- `log_file`
- `append_log`
- `log_fields`
- `timing_file`
- `append_trajectory`
- `trajectory_interval`
- `enable_stopcar`
- `stopcar_file`
- `stop_check_interval`
- `stop_on_custom_condition`
- `stop_condition_module`
- `stop_condition_function`
- `create_stopcar_on_custom_condition`
- `thermostat`
- `temperature_k`
- `time_step_fs`
- `steps`
- `tdamp_fs`
- `tchain`
- `tloop`
- `friction`
- `taut_fs`
- `initialize_velocities`
- `remove_translation`
- `remove_rotation`
- `log_interval`

Supported thermostats:

- `nose-hoover`
- `langevin`
- `berendsen`
- `bussi`

`nose-hoover` is the default.

### NVE Molecular Dynamics

```bash
python3 scripts/run_nve.py
python3 scripts/run_nve.py inputs/md/nve.yaml
python3 scripts/run_nve.py inputs/md/nve_short_test.yaml
```

Key YAML options:

- `structure_file`
- `structure_files`
- `structure_glob`
- `selected_models`
- `device`
- `default_dtype`
- `output_name`
- `trajectory_file`
- `trajectory_format`
- `final_structure_file`
- `final_structure_format`
- `save_final_structure`
- `log_file`
- `append_log`
- `log_fields`
- `timing_file`
- `append_trajectory`
- `trajectory_interval`
- `enable_stopcar`
- `stopcar_file`
- `stop_check_interval`
- `stop_on_custom_condition`
- `stop_condition_module`
- `stop_condition_function`
- `create_stopcar_on_custom_condition`
- `time_step_fs`
- `steps`
- `log_interval`

## MD YAML Keyword Reference

The MD YAML files share a common set of keywords across `optimize`, `nvt`, and `nve`. Task-specific controls are listed separately below.

### Common Structure Selection Keywords

- `structure_file`
  Single input structure. This can be a filename inside `inputs/md/`, a path relative to the project root, or an absolute path.
- `structure_files`
  Explicit YAML list of multiple input structures processed sequentially in one run.
- `structure_glob`
  Glob pattern used to select several input structures automatically, including subdirectories such as `100traj_AIMD/batch_1/POSCAR-*`.
- `selected_models`
  YAML list of model filenames to use from `model/`. Use `null` to run all detected models.

Only one structure selection mode is needed. If several are provided, the resolved file set is combined and de-duplicated.

### Common Runtime and I/O Keywords

- `device`
  Execution device. Supported values: `cpu`, `cuda`.
- `default_dtype`
  Floating-point precision passed to the calculator. Typical values: `float32`, `float64`.
- `output_name`
  Name of the output subdirectory under `outputs/md/`.
- `trajectory_file`
  Output trajectory filename.
- `trajectory_format`
  Trajectory format written during the run. Recommended values: `extxyz`, `traj`.
- `final_structure_file`
  Filename for the last saved structure.
- `final_structure_format`
  Format for the final saved structure. Use `traj` when restart-ready velocities must be preserved reliably.
- `save_final_structure`
  If `true`, write the last structure at the end of the run.
- `log_file`
  Runtime log filename.
- `append_log`
  If `true`, append to an existing runtime log instead of overwriting it.
- `timing_file`
  Filename for the wall-time summary YAML. Default output is `timing_summary.yaml`.
- `append_trajectory`
  If `true`, append to an existing trajectory file when the chosen writer supports it.
- `trajectory_interval`
  Write one trajectory frame every `N` MD or optimization steps.
- `log_interval`
  Write one runtime log entry every `N` MD or optimization steps.

### Runtime Log Keywords

- `log_fields`
  YAML list defining the columns written to `run.log`.
- `monitored_distances`
  YAML list of extra geometric observables that can be referenced inside `log_fields`.

Supported built-in `log_fields` values are:

- `step`
- `time_fs`
- `temperature_K`
- `potential_energy_eV`
- `kinetic_energy_eV`
- `total_energy_eV`
- `natoms`
- `max_force_eV_A`

Each entry in `monitored_distances` must define:

- `name`
  Column name exposed to `log_fields`.
- `kind`
  Supported values: `atom_distance`, `com_z_distance`, `average_z_distance`.

Additional required keys depend on `kind`:

- `atom_distance`
  Requires `atom_indices` and optional `index_base`.
- `com_z_distance`
  Requires `group_a_indices`, `group_b_indices`, and optional `index_base`.
- `average_z_distance`
  Requires `group_a_indices`, `group_b_indices`, and optional `index_base`.

Indexing notes:

- `index_base: 1` means human-style indexing, consistent with VASP or LAMMPS atom numbering.
- `index_base: 0` means Python-style indexing.

### Early Stop Keywords

- `enable_stopcar`
  Enable periodic checks for a `STOPCAR` file in the run output directory.
- `stopcar_file`
  Name of the stop file to monitor.
- `stop_check_interval`
  Check the stop condition every `N` steps.
- `stop_on_custom_condition`
  Enable a Python callback for early stopping.
- `stop_condition_module`
  Python module name containing the callback function.
- `stop_condition_function`
  Function name to call inside that module.
- `create_stopcar_on_custom_condition`
  If `true`, write a `STOPCAR` file automatically when the custom condition triggers.

If `stop_on_custom_condition: true`, both `stop_condition_module` and `stop_condition_function` must be set.

### Built-in Gas-Surface Scattering Stop Keywords

These options are used only with:

- `stop_condition_module: stop_conditions`
- `stop_condition_function: gas_surface_scattering_reflection_stop`

Keywords:

- `scattering_molecule_indices`
  Atom indices defining the incident molecule.
- `scattering_surface_top_indices`
  Atom indices defining the top reference layer of the slab.
- `scattering_index_base`
  Index convention for the two lists above. Use `1` for VASP/LAMMPS-style numbering.
- `scattering_activation_distance_a`
  The stop logic arms only after the molecule-surface distance first becomes smaller than this value.
- `scattering_stop_distance_a`
  After the stop logic is armed, the run stops when the molecule moves away and exceeds this value.
- `scattering_surface_z_mode`
  Surface height definition. Supported values currently target the top-layer reference, with `average` as the standard choice.
- `scattering_require_outgoing`
  If `true`, the stop logic also requires the molecule to move outward before stopping.
- `scattering_distance_hysteresis_a`
  Small numerical tolerance used to avoid noisy threshold triggering.

### Optimize-Specific Keywords

- `optimizer`
  ASE optimizer name. Supported values in the current script: `bfgs`, `lbfgs`, `fire`.
- `fmax`
  Force convergence threshold in `eV/A`.
- `steps`
  Maximum number of optimization steps.
- `optimizer_log_file`
  Optimizer text log written by ASE.
- `optimizer_restart_file`
  Optimizer restart data written by ASE when supported by the optimizer.

### NVT-Specific Keywords

- `thermostat`
  Supported values: `nose-hoover`, `langevin`, `berendsen`, `bussi`.
- `temperature_k`
  Target temperature in kelvin.
- `time_step_fs`
  Integration timestep in femtoseconds.
- `steps`
  Number of MD integration steps.
- `tdamp_fs`
  Nose-Hoover damping time in femtoseconds.
- `tchain`
  Number of Nose-Hoover chain thermostats.
- `tloop`
  Number of thermostat sub-iterations for Nose-Hoover.
- `friction`
  Langevin friction coefficient.
- `taut_fs`
  Thermostat relaxation time used by Berendsen and Bussi.
- `initialize_velocities`
  If `true`, initialize velocities from the requested temperature before the run.
- `remove_translation`
  Remove center-of-mass translation after velocity initialization.
- `remove_rotation`
  Remove overall rotation after velocity initialization.

`nose-hoover` is the default thermostat.

### NVE-Specific Keywords

- `time_step_fs`
  Integration timestep in femtoseconds.
- `steps`
  Number of MD integration steps.

`run_nve.py` expects input structures that already contain velocities if a physically meaningful NVE trajectory is required.

## Outputs

Correlation outputs are written to:

```text
outputs/correlation/
```

MD outputs are written to:

```text
outputs/md/<output_name>/<structure_name>/<model_name>/
```

Depending on the selected task and YAML settings, the output directory may contain:

- trajectory files
- final structure files
- `run.log`
- `optimizer.log`
- `optimizer.restart`
- `run_settings.yaml`
- `timing_summary.yaml`
- `STOPCAR`

`timing_summary.yaml` includes:

- total wall time
- requested number of steps
- completed number of steps
- average wall time per step

For NVT and NVE, it also includes simulated time information.

Final structures are written only if `save_final_structure: true`.

For restartable outputs with velocities, `traj` and `extxyz` are the recommended formats for `final_structure_format`.

## Logging and Runtime Information

At launch, each script prints:

- the selected task
- the YAML input file used
- the output root directory
- the selected device and dtype
- the selected models
- the selected structures

For each individual run, the scripts print:

- the structure being processed
- the model being processed
- the corresponding output directory
- the final timing summary

The resolved YAML file path is also stored in `run_settings.yaml`.

The runtime log file is configurable and can include:

- step number
- simulated time
- temperature
- potential energy
- kinetic energy
- total energy
- maximum force
- user-defined monitored distances

Example:

```yaml
log_file: run.log
append_log: false
log_fields:
  - step
  - time_fs
  - temperature_K
  - potential_energy_eV
  - kinetic_energy_eV
  - total_energy_eV
  - NO_bond

monitored_distances:
  - name: NO_bond
    kind: atom_distance
    atom_indices: [1, 2]
    index_base: 1
```

Supported monitored distance types are:

- `atom_distance`
- `com_z_distance`
- `average_z_distance`

More detailed logging and very frequent write intervals increase I/O and can reduce overall performance, especially for short timesteps or large systems. In practice, the overhead is usually modest when `log_interval` and `trajectory_interval` are not too small, but very aggressive logging can noticeably slow a run.

## Early Stop Control

All three calculation scripts support early termination through a `STOPCAR` file.

If `enable_stopcar: true`, the code checks for `STOPCAR` in the run output directory at the interval defined by `stop_check_interval`. If the file is found, the calculation stops cleanly before the requested number of steps is reached.

The stop status is recorded in `timing_summary.yaml`, including:

- `stopped_early`
- `stop_reason`

An optional external Python routine can also trigger early termination:

- `stop_on_custom_condition: true`
- `stop_condition_module: my_stop_conditions`
- `stop_condition_function: should_stop`

The callable is expected to return either:

- `True` or `False`
- `(True, "reason")` or `(False, None)`

When `create_stopcar_on_custom_condition: true`, a custom stop condition automatically writes `STOPCAR` before stopping the run.

### Built-in Gas-Surface Scattering Stop Condition

The repository includes a built-in routine for gas-surface scattering trajectories:

- `stop_condition_module: stop_conditions`
- `stop_condition_function: gas_surface_scattering_reflection_stop`

This routine is designed for cases where a molecule is launched toward a slab and the trajectory should stop only after the molecule has approached the surface and then moved away again.

Required YAML settings:

- `scattering_molecule_indices`
- `scattering_surface_top_indices`
- `scattering_activation_distance_a`
- `scattering_stop_distance_a`

Optional settings:

- `scattering_index_base`
- `scattering_surface_z_mode`
- `scattering_require_outgoing`
- `scattering_distance_hysteresis_a`

The logic is:

1. compute the z-distance between the molecule center of mass and the average z-position of the selected top surface layer
2. arm the stop criterion only after the molecule has first crossed below `scattering_activation_distance_a`
3. stop the trajectory only when the molecule later moves away and exceeds `scattering_stop_distance_a`

This avoids stopping immediately at the beginning of the run when the initial molecule-surface distance is already larger than the final stop distance.

Example:

```yaml
stop_on_custom_condition: true
stop_condition_module: stop_conditions
stop_condition_function: gas_surface_scattering_reflection_stop
create_stopcar_on_custom_condition: true

scattering_molecule_indices: [1, 2]
scattering_surface_top_indices: [259, 260, 261, 262]
scattering_index_base: 1
scattering_activation_distance_a: 6.0
scattering_stop_distance_a: 8.0
scattering_surface_z_mode: average
scattering_require_outgoing: true
scattering_distance_hysteresis_a: 1.0e-3
```

## CPU and GPU Execution

The scripts support:

- `device: cpu`
- `device: cuda`

For CPU runs, PyTorch may use multiple CPU threads depending on the runtime environment. This repository does not currently expose thread control directly in the YAML, so multi-thread CPU usage is typically controlled through environment variables such as:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

This provides multi-threaded CPU execution within a single process. The scripts do not currently implement MPI or multi-node parallelism for a single trajectory.

## Cluster Usage

A typical cluster workflow is:

1. copy the repository to the cluster
2. create the Python environment on the cluster
3. prepare one or more YAML input files in `inputs/md/`
4. submit the desired script through a batch job

Example NVT YAML:

```yaml
structure_glob: "POSCAR-*"
selected_models: null
device: cpu
output_name: nvt_batch
thermostat: nose-hoover
temperature_k: 300.0
time_step_fs: 1.0
steps: 5000
tdamp_fs: 100.0
trajectory_file: trajectory.extxyz
trajectory_format: extxyz
final_structure_file: final_structure.extxyz
final_structure_format: extxyz
trajectory_interval: 10
log_interval: 10
```

Example Slurm job:

```bash
#!/bin/bash
#SBATCH --job-name=mace-nvt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=cpu
#SBATCH --output=slurm-%j.out

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /path/to/MACE_fine_tuning
source .venv/bin/activate
python3 scripts/run_nvt.py inputs/md/nvt_batch_300K.yaml
```

For GPU jobs:

- install a CUDA-compatible PyTorch build on the target system
- set `device: cuda` in the YAML file
- submit the job to a GPU partition

## NVE Requirement

`run_nve.py` requires initial velocities to already be present in the input structure.

If the input file does not contain velocities, a typical workflow is:

1. run optimization or NVT first
2. restart NVE from a structure or trajectory frame that already contains velocities

## Dependencies

The project dependencies are listed in [requirements.txt](/Users/samuel/Desktop/postdoc_PhLAM/codes/MACE_fine_tuning/requirements.txt), including:

- `torch`
- `mace-torch`
- `cuequivariance`
- `cuequivariance-torch`

`cuequivariance` support may accelerate some MACE calculations depending on the installed PyTorch/MACE stack and the hardware configuration.
