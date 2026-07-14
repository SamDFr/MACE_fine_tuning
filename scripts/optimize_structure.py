from __future__ import annotations

from _md_shared import (
    attach_trajectory_and_log,
    build_optimizer,
    load_run_config,
    load_atoms,
    load_calculator,
    model_output_dir,
    prepare_named_output_dir,
    save_final_structure,
    select_model_files,
    select_structure_files,
    write_run_settings,
)


def main() -> None:
    config = load_run_config("optimize")
    model_paths = select_model_files(config)
    structure_paths = select_structure_files(config)
    output_root = prepare_named_output_dir("optimize", config)

    device = config.get("device", "cpu")
    default_dtype = config.get("default_dtype", "float32")
    optimizer = config.get("optimizer", "fire")
    fmax = float(config.get("fmax", 0.03))
    steps = int(config.get("steps", 300))

    for structure_path in structure_paths:
        print(f"Structure used: {structure_path}")
        for model_path in model_paths:
            output_dir = model_output_dir(output_root, model_path, structure_path=structure_path)
            atoms = load_atoms(structure_path)
            atoms.calc = load_calculator(model_path, device=device, default_dtype=default_dtype)

            dyn = build_optimizer(optimizer, atoms, output_dir, config=config)
            attach_trajectory_and_log(dyn, atoms, output_dir, config=config)
            dyn.run(fmax=fmax, steps=steps)
            save_final_structure(atoms, output_dir, config=config)
            write_run_settings(output_dir, config, model_path, structure_path)

            print(f"Model used: {model_path}")
            print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
