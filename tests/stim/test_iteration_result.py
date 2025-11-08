import numpy as np
import pathlib
import sinter
import stim
import tempfile
import sys
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from relay_bp.stim import (
    SinterDecoder_RelayBP,
    sinter_decoders,
    CheckMatrices,
)

from testdata import (
    get_test_circuit,
    get_all_test_circuits,
    filter_detectors_by_basis,
)




def test_sinter_saves_results_to_csv():
    """Test sinter.collect saves the result including iterations"""
    circuit = get_test_circuit("bicycle_bivariate_18_4_3_memory_Z", 0.001)
    tasks = [sinter.Task(circuit=circuit)]

    # Set directory and filepath for saving test results
    output_dir = pathlib.Path("tests/test_outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_path = output_dir / "sinter_results_test.csv"

    decoder_params = dict(
        gamma0=0.1,
        pre_iter=5,
        num_sets=1,
        set_max_iter=5,
        gamma_dist_interval=[-0.24, 0.66],
        stop_nconv=1,
        get_detail=True
        )
    decoders = sinter_decoders(
        **decoder_params
    )

    sinter.collect(
        num_workers=3,
        max_shots=1000,
        tasks=tasks,
        decoders=["relay-bp"],
        custom_decoders=decoders,
        save_resume_filepath=csv_output_path,
    )

    print(f"\n--> Test results saved to: {csv_output_path}")


if __name__ == "__main__":
    test_sinter_saves_results_to_csv()

