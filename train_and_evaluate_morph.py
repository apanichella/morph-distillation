import os
import argparse
import subprocess


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Train and evaluate MORPH models.")
    parser.add_argument("--task", type=str, required=True,
                        choices=["clone-detection", "vulnerability-detection"],
                        help="The task to run: clone-detection or vulnerability-detection.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["codebert", "graphcodebert"],
                        help="The model to use: codebert or graphcodebert.")
    args = parser.parse_args()

    # Hardcoded paths for scripts and surrogate data (absolute paths)
    paths = {
        ("clone-detection", "codebert"): {
            "script": "/root/Morph/CodeBERT/Clone-Detection/Morph/many_objective.py",
            "data": "/root/Morph/CodeBERT/Clone-Detection/Morph/surrogate_data_metamorphic.csv",
        },
        ("clone-detection", "graphcodebert"): {
            "script": "/root/Morph/GraphCodeBERT/Clone-Detection/Morph/many_objective.py",
            "data": "/root/Morph/GraphCodeBERT/Clone-Detection/Morph/surrogate_data_metamorphic.csv",
        },
        ("vulnerability-detection", "codebert"): {
            "script": "/root/Morph/CodeBERT/Vulnerability-Detection/Morph/many_objective.py",
            "data": "/root/Morph/CodeBERT/Vulnerability-Detection/Morph/surrogate_data_metamorphic.csv",
        },
        ("vulnerability-detection", "graphcodebert"): {
            "script": "/root/Morph/GraphCodeBERT/Vulnerability-Detection/Morph/many_objective.py",
            "data": "/root/Morph/GraphCodeBERT/Vulnerability-Detection/Morph/surrogate_data_metamorphic.csv",
        },
    }

    # Get the paths based on the task and model
    key = (args.task, args.model)
    if key not in paths:
        print("Invalid combination of task and model.")
        exit(1)

    script_path = paths[key]["script"]
    surrogate_data_path = paths[key]["data"]

    # Check if the script exists
    if not os.path.exists(script_path):
        print(f"Error: Script file not found at {script_path}. Please check the path.")
        exit(1)

    # Check if the surrogate data file exists
    if not os.path.exists(surrogate_data_path):
        print(f"Error: Required file 'surrogate_data_metamorphic.csv' not found at {surrogate_data_path}.")
        exit(1)

    # Change to the directory of the script
    script_dir = os.path.dirname(script_path)
    os.chdir(script_dir)

    # Check for CUDA availability
    cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if cuda_devices is None:
        try:
            # Detect GPUs
            n_gpus = int(subprocess.check_output(
                "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l",
                shell=True
            ).strip())
            if n_gpus > 0:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
        except Exception:
            print("CUDA is not available. Running on CPU.")

    # Prepare the command to run the script
    command = f"python3 {os.path.basename(script_path)}"

    # Notify the user about the execution environment
    if os.getenv("CUDA_VISIBLE_DEVICES"):
        print(f"Using GPU(s): {os.getenv('CUDA_VISIBLE_DEVICES')}")
    else:
        print("Running on CPU.")

    # Run the selected script
    print(f"Running: {command}")
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()