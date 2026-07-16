from __future__ import annotations

from _md_shared import (
    attach_trajectory_and_log,
    attach_stop_monitor,
    build_timing_summary,
    load_run_config,
    load_atoms,
    load_calculator,
    measure_run,
    model_output_dir,
    parse_config_cli,
    print_model_header,
    print_run_header,
    print_structure_header,
    print_timing_summary,
    prepare_named_output_dir,
    save_final_structure,
    select_model_files,
    select_structure_files,
    setup_nve,
    write_timing_summary,
    write_run_settings,
)


def main() -> None:
    config_path = parse_config_cli("nve")
    config, resolved_config_path = load_run_config("nve", config_file=str(config_path))
    config["_task_name"] = "nve"
    model_paths = select_model_files(config)
    structure_paths = select_structure_files(config)
    output_root = prepare_named_output_dir("nve", config)
    print_run_header("nve", resolved_config_path, config, model_paths, structure_paths, output_root)

    device = config.get("device", "cpu")
    default_dtype = config.get("default_dtype", "float32")
    time_step_fs = float(config.get("time_step_fs", 1.0))
    steps = int(config.get("steps", 2000))

    for structure_index, structure_path in enumerate(structure_paths, start=1):
        print_structure_header("nve", structure_path, structure_index, len(structure_paths))
        for model_index, model_path in enumerate(model_paths, start=1):
            output_dir = model_output_dir(output_root, model_path, structure_path=structure_path)
            print_model_header("nve", model_path, model_index, len(model_paths), output_dir)
            atoms = load_atoms(structure_path)
            atoms.calc = load_calculator(model_path, device=device, default_dtype=default_dtype)

            dyn = setup_nve(atoms, time_step_fs=time_step_fs)
            attach_trajectory_and_log(dyn, atoms, output_dir, config=config)
            stopcar_path = attach_stop_monitor(dyn, atoms, output_dir, config=config)
            elapsed_seconds, started_at, finished_at, stopped_early, stop_reason = measure_run(
                lambda: dyn.run(steps)
            )
            save_final_structure(atoms, output_dir, config=config)
            completed_steps = int(getattr(dyn, "nsteps", steps))
            timing_summary = build_timing_summary(
                task="nve",
                config=config,
                requested_steps=steps,
                completed_steps=completed_steps,
                elapsed_seconds=elapsed_seconds,
                started_at=started_at,
                finished_at=finished_at,
                stopped_early=stopped_early,
                stop_reason=stop_reason,
            )
            write_timing_summary(output_dir, config, timing_summary)
            write_run_settings(
                output_dir,
                config,
                model_path,
                structure_path,
                config_path=resolved_config_path,
                timing_summary=timing_summary,
            )

            print(f"Completed calculation for structure: {structure_path.name}")
            print(f"Completed calculation for model: {model_path.name}")
            print(f"Outputs written to: {output_dir}")
            print(f"STOPCAR path: {stopcar_path}")
            print_timing_summary(timing_summary)


if __name__ == "__main__":
    main()
