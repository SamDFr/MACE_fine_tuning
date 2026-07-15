# Projet simple pour MACE fine-tuné

Ce dépôt est maintenant organisé de façon simple.

Tu poses:

- tes modèles dans `model/`
- tes fichiers d'entrée dans `inputs/`
- les résultats sortiront dans `outputs/`

## Structure

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

## Ce que tu dois faire

### 1. Modèles

Mets tes modèles MACE fine-tunés dans `model/`.

Exemple:

```text
model/
├── seed_1.model
├── seed_2.model
└── seed_3.model
```

### 2. Données pour corrélations

Mets un ou plusieurs fichiers test dans:

```text
inputs/correlation/
```

Le notebook de corrélation détecte automatiquement s'il y a plusieurs fichiers à analyser.

Formats actuellement pris en charge automatiquement:

- `vasprun*.xml`
- `*.traj`
- `*.extxyz`
- `*.xyz`

Pour les `vasprun.xml` et `traj` ASE, le notebook relit automatiquement les énergies et forces de référence si elles sont stockées dans le calculateur attaché.

### 3. Données pour MD

Mets la structure à utiliser dans:

```text
inputs/md/
```

Exemples:

- `inputs/md/POSCAR`
- `inputs/md/POSCAR-1`
- `inputs/md/structure.xyz`
- `inputs/md/structure.traj`
- `inputs/md/structure.extxyz`
- `inputs/md/vasprun.xml`

Les scripts MD lisent maintenant leurs paramètres de simulation depuis des fichiers YAML dans `inputs/md/`.
Ils peuvent aussi traiter plusieurs structures en batch.

Pour un batch de plusieurs structures `POSCAR-X`, tu peux utiliser par exemple:

```yaml
structure_glob: "POSCAR-*"
```

ou une liste explicite:

```yaml
structure_files:
  - POSCAR-160
  - POSCAR-161
  - POSCAR-162
```

## Notebooks

### Corrélations

Ouvre:

```text
notebooks/01_correlation_plots.ipynb
```

Ce notebook:

- cherche automatiquement tous les modèles dans `model/`
- cherche automatiquement un ou plusieurs fichiers dans `inputs/correlation/`
- écrit les résultats dans `outputs/correlation/`
- permet aussi de limiter l'analyse à certains fichiers
- permet aussi d'échantillonner les frames avec un stride

Sorties:

- un sous-dossier par modèle dans `outputs/correlation/`
- un résumé global comparant les modèles
- une figure de corrélation énergie par modèle
- une figure de corrélation sur la norme des forces par modèle
- une figure `Fx/Fy/Fz` par espèce atomique et par modèle
- une figure d'analyse de direction des forces par modèle

Réglages utiles en haut du notebook:

- `SELECTED_FILES = None` pour traiter tous les fichiers détectés
- `SELECTED_FILES = ['file1.xml', 'file2.traj']` pour n'en traiter qu'une partie
- `FRAME_STRIDE = 1` pour garder toutes les structures
- `FRAME_STRIDE = 10` pour garder 1 frame sur 10

### Vérification locale avant MD

Ouvre:

```text
notebooks/02_check_md_inputs.ipynb
```

Ce notebook permet de:

- vérifier quels modèles ont été trouvés
- vérifier quel fichier structure a été trouvé
- vérifier si des vitesses sont présentes pour NVE
- voir rapidement la structure lue par ASE

## Scripts Python

### Optimisation

```bash
python3 scripts/optimize_structure.py
```

Le script lit automatiquement:

- `inputs/md/optimize.yaml`

Tu modifies ce fichier pour régler:

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
- `log_file`
- `timing_file`
- `append_trajectory`
- `trajectory_interval`
- `optimizer_log_file`
- `optimizer_restart_file`
- `optimizer`
- `fmax`
- `steps`
- `log_interval`

Pour les sorties, tu peux par exemple utiliser:

- `trajectory_format: extxyz`
- `trajectory_format: traj`
- `trajectory_format: xyz`
- `trajectory_format: lammps-dump-text`

Sorties dans:

```text
outputs/md/optimize/
```

avec un sous-dossier par modèle
et par structure si plusieurs structures sont lancées

Chaque run écrit aussi:

- `run_settings.yaml`
- `timing_summary.yaml`

Le fichier `timing_summary.yaml` contient notamment:

- le temps mur total du calcul
- le nombre de pas demandés
- le nombre de pas effectivement réalisés
- le temps mur moyen par pas

### NVT

```bash
python3 scripts/run_nvt.py
```

Le script lit automatiquement:

- `inputs/md/nvt.yaml`

Tu modifies ce fichier pour régler:

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
- `log_file`
- `timing_file`
- `append_trajectory`
- `trajectory_interval`
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

Chaque run écrit aussi `timing_summary.yaml` avec le temps mur total et le temps moyen par pas MD.

Thermostats currently supported for NVT:

- `nose-hoover` (default)
- `langevin`
- `berendsen`
- `bussi`

Pour les sorties, tu peux par exemple utiliser:

- `trajectory_format: extxyz`
- `trajectory_format: traj`
- `trajectory_format: xyz`
- `trajectory_format: lammps-dump-text`

Sorties dans:

```text
outputs/md/nvt/
```

avec un sous-dossier par modèle
et par structure si plusieurs structures sont lancées

### NVE

```bash
python3 scripts/run_nve.py
```

Le script lit automatiquement:

- `inputs/md/nve.yaml`

Tu modifies ce fichier pour régler:

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
- `log_file`
- `timing_file`
- `append_trajectory`
- `trajectory_interval`
- `time_step_fs`
- `steps`
- `log_interval`

Chaque run écrit aussi `timing_summary.yaml` avec le temps mur total et le temps moyen par pas MD.

Pour les sorties, tu peux par exemple utiliser:

- `trajectory_format: extxyz`
- `trajectory_format: traj`
- `trajectory_format: xyz`
- `trajectory_format: lammps-dump-text`

Sorties dans:

```text
outputs/md/nve/
```

avec un sous-dossier par modèle
et par structure si plusieurs structures sont lancées

## Dépendances minimales

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si tu veux lancer les notebooks:

```bash
source .venv/bin/activate
jupyter notebook
```

## Cluster

Le plus simple sur cluster est:

1. copier ce dépôt sur le cluster
2. créer l'environnement Python sur le cluster
3. modifier `inputs/md/optimize.yaml`, `inputs/md/nvt.yaml` ou `inputs/md/nve.yaml`
4. lancer le script Python voulu depuis un job batch

Exemple CPU avec plusieurs `POSCAR-X`:

Dans le YAML:

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

Exemple `job.slurm`:

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

cd /path/to/MACE_fine_tuning
source .venv/bin/activate
python3 scripts/run_nvt.py
```

Pour GPU:

- installe une version de `torch` compatible CUDA sur le cluster
- mets `device: cuda` dans le YAML
- soumets le job sur une partition GPU

Les sorties seront rangées comme:

```text
outputs/md/<output_name>/<structure_name>/<model_name>/
```

## Remarque importante pour NVE

Le script `run_nve.py` demande des vitesses déjà présentes dans le fichier d'entrée.

Si ton fichier n'a pas de vitesses, il faut commencer par:

- une optimisation
- ou une NVT

puis relancer NVE à partir d'une structure qui contient des vitesses.
