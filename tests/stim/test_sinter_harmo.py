import stim
import sinter
import numpy as np
from relay_bp.stim import sinter_decoders
import pathlib
import matplotlib.pyplot as plt
import sys 
import os
test_data_path = pathlib.Path(__file__).parent.parent
assert test_data_path.exists()
sys.path.append(str(test_data_path))
from testdata import (
    get_test_circuit,
    get_all_test_circuits,
    filter_detectors_by_basis,
)
# def generate_example_tasks():
#     # Using a small code for faster testing
#     for p in np.logspace(-3, -2, 5):
#         for d in [3, 5, 7, 9]:
#             yield sinter.Task(
#                 circuit=stim.Circuit.generated(
#                     rounds=d,
#                     distance=d,
#                     after_clifford_depolarization=p,
#                     before_measure_flip_probability=p,
#                     code_task=f"surface_code:rotated_memory_x",
#                 ),
#                 json_metadata={"p": p, "d": d},
#             )   

# def test_sinter_harmonized_bp_on_example_tasks():
#     decoder_params = dict(
#         gamma0=0.1,
#         pre_iter=30,
#         num_sets=5,
#         set_max_iter=20,
#         gamma_dist_interval=[-0.24, 0.66],
#         stop_nconv=1,
#         get_detail=True,
#         ensemble_size=16,
#         use_automorphism=True,
#         perturbation_min=0.1,
#         perturbation_max=0.1,
#         selection_strategy= "MostLikely"
#         )
#     decoders = sinter_decoders(
#         **decoder_params
#     )

#     # --- CSV Output Setup ---
#     output_dir = pathlib.Path("./test_output")
#     output_dir.mkdir(parents=True, exist_ok=True)
#     # Use a unique filename for each test case to prevent conflicts
#     csv_output_path = output_dir / f"sinter_harmonizedbp_test.csv"

#     if csv_output_path.exists():
#         csv_output_path.unlink()  # Ensure a fresh test

#     samples = sinter.collect(
#         num_workers=4,  # Use 1 worker for easier debugging if bliss fails
#         max_shots=10000,   # Fewer shots for faster test execution
#         tasks=generate_example_tasks(),
#         decoders=["harmonized-bp", "relay-bp"],
#         custom_decoders=decoders,
#         save_resume_filepath=csv_output_path,
#         print_progress=True,  
#     )


#     # Render a matplotlib plot of the data.
#     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#     sinter.plot_error_rate(
#         ax=ax,
#         stats=samples,
#         group_func=lambda stat: f"""{stat.json_metadata["d"]}, decoder={stat.decoder}""",
#         x_func=lambda stat: stat.json_metadata["p"],
#     )
#     ax.loglog()
#     ax.grid()
#     ax.set_title(f"Logical Error Rate vs Physical Error Rate for Optimized Relay Parameters")
#     ax.set_ylabel("Logical Error Probability (per round/qubit)")
#     ax.set_xlabel("Physical Error Rate")
#     ax.legend()

#     plt.show()

def generate_sinter_tasks(decoders, decoder_params):

    n = 144
    k = 12
    d = 12
    basis = "Z"
    circuit_strs = [f"bicycle_bivariate_{n}_{k}_{d}_memory_{basis}"]

    error_rates = np.linspace(0.002, 0.005, 4)
    for circuit_str in circuit_strs: 
        for error_rate in error_rates:
            circuit = get_test_circuit(circuit=circuit_str, distance=d, rounds=d, error_rate=error_rate,)
            for XYZ in [False]:
                if not XYZ:
                    circuit = filter_detectors_by_basis(circuit, basis)
                dem = circuit.detector_error_model()
                for decoder in decoders.keys():
                    yield sinter.Task(
                        circuit=circuit,
                        detector_error_model=dem,
                        decoder=decoder,
                        json_metadata={
                            "circuit": circuit_str,
                            "p": error_rate,
                            "n": n,
                            "k": k,
                            "d": d,
                            "r": d,
                            "xyz": XYZ,
                            "decoder_params": decoder_params,
                        },
                        collection_options=sinter.CollectionOptions(max_shots=100000)
                    )

    

def test_sinter_harmonized_bp_on_bbcode():
    n = 144
    k = 12
    d = 12
    basis = "Z"
    circuit_strs = [f"bicycle_bivariate_{n}_{k}_{d}_memory_{basis}"]

    decoder_params = dict(
        gamma0=0.1,
        pre_iter=30,
        num_sets=5,
        set_max_iter=20,
        gamma_dist_interval=[-0.24, 0.66],
        stop_nconv=1,
        get_detail=True,
        ensemble_size=16,
        use_automorphism=True,
        perturbation_min=0.1,
        perturbation_max=0.1,
        selection_strategy= "MostLikely"
        )
    decoders = sinter_decoders(
        **decoder_params
    )
    # --- CSV Output Setup ---
    output_dir = pathlib.Path("./test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use a unique filename for each test case to prevent conflicts
    csv_output_path = output_dir / f"sinter_harmonizedbp_test.csv"

    if csv_output_path.exists():
        csv_output_path.unlink()  # Ensure a fresh test

    samples = sinter.collect(
        num_workers=96, 
        tasks=generate_sinter_tasks(decoders, decoder_params),
        decoders=["harmonized-bp", "relay-bp", "mem-bp"],
        custom_decoders=decoders,
        save_resume_filepath=csv_output_path,
        print_progress=False,  
    )


    # Render a matplotlib plot of the data.
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f"""{stat.json_metadata["circuit"]}, decoder={stat.decoder}{", XYZ" if stat.json_metadata["xyz"] else ""}""",
        x_func=lambda stat: stat.json_metadata["p"],
        failure_units_per_shot_func=lambda stats: stats.json_metadata["d"],
        failure_values_func=lambda stats: stats.json_metadata['k'],
        filter_func=lambda stat: stat.json_metadata["decoder_params"] == decoder_params and stat.json_metadata["circuit"] in circuit_strs
    )
    ax.loglog()
    ax.grid()
    ax.set_title(f"Logical Error Rate vs Physical Error Rate for Optimized Relay Parameters")
    ax.set_ylabel("Logical Error Probability (per round/qubit)")
    ax.set_xlabel("Physical Error Rate")
    ax.legend()
    plt.savefig("sinter_harmonized_bp_on_bbcode.png")

    plt.show()

if __name__ == "__main__":
    test_sinter_harmonized_bp_on_bbcode()