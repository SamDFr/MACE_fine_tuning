from __future__ import annotations

from _md_shared import (
    attach_trajectory_and_log,
    attach_stop_monitor,
    build_optimizer,
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
    run_optimizer_with_consecutive_convergence,
    save_final_structure,
    select_model_files,
    select_structure_files,
    write_timing_summary,
    write_run_settings,
)


def main() -> None:
    config_path = parse_config_cli("optimize")
    config, resolved_config_path = load_run_config("optimize", config_file=str(config_path))
    config["_task_name"] = "optimize"
    model_paths = select_model_files(config)
    structure_paths = select_structure_files(config)
    output_root = prepare_named_output_dir("optimize", config)
    print_run_header("optimize", resolved_config_path, config, model_paths, structure_paths, output_root)

    device = config.get("device", "cpu")
    default_dtype = config.get("default_dtype", "float32")
    optimizer = config.get("optimizer", "fire")
    fmax = float(config.get("fmax", 0.03))
    steps = int(config.get("steps", 300))
    fmax_consecutive = int(config.get("fmax_consecutive", 1))
    energy_tolerance_eV = config.get("energy_tolerance_eV")
    if energy_tolerance_eV is not None:
        energy_tolerance_eV = float(energy_tolerance_eV)
    max_total_displacement_a = config.get("max_total_displacement_a")
    if max_total_displacement_a is not None:
        max_total_displacement_a = float(max_total_displacement_a)

    for structure_index, structure_path in enumerate(structure_paths, start=1):
        print_structure_header("optimize", structure_path, structure_index, len(structure_paths))
        for model_index, model_path in enumerate(model_paths, start=1):
            output_dir = model_output_dir(output_root, model_path, structure_path=structure_path)
            print_model_header("optimize", model_path, model_index, len(model_paths), output_dir)
            atoms = load_atoms(structure_path)
            atoms.calc = load_calculator(model_path, device=device, default_dtype=default_dtype)

            dyn = build_optimizer(optimizer, atoms, output_dir, config=config)
            attach_trajectory_and_log(dyn, atoms, output_dir, config=config)
            stopcar_path = attach_stop_monitor(dyn, atoms, output_dir, config=config)
            optimization_result = {}

            def run_optimization() -> None:
                optimization_result.update(
                    run_optimizer_with_consecutive_convergence(
                        dyn,
                        atoms,
                        fmax=fmax,
                        steps=steps,
                        consecutive=fmax_consecutive,
                        energy_tolerance_eV=energy_tolerance_eV,
                        max_total_displacement_a=max_total_displacement_a,
                    )
                )

            elapsed_seconds, started_at, finished_at, stopped_early, stop_reason = measure_run(
                run_optimization
            )
            save_final_structure(atoms, output_dir, config=config)
            completed_steps = int(getattr(dyn, "nsteps", steps))
            timing_summary = build_timing_summary(
                task="optimize",
                config=config,
                requested_steps=steps,
                completed_steps=completed_steps,
                elapsed_seconds=elapsed_seconds,
                started_at=started_at,
                finished_at=finished_at,
                stopped_early=stopped_early,
                stop_reason=stop_reason,
            )
            timing_summary["optimization"] = optimization_result
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
            if optimization_result:
                print(f"Optimization converged: {optimization_result['converged']}")
                print(
                    "Final maximum unconstrained force: "
                    f"{optimization_result['final_max_force_eV_A']:.6f} eV/A"
                )
                print(
                    "Consecutive force checks: "
                    f"{optimization_result['fmax_consecutive_reached']}/"
                    f"{optimization_result['fmax_consecutive_required']}"
                )
            print_timing_summary(timing_summary)


if __name__ == "__main__":
    main()
