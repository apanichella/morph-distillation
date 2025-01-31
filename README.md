#MORPH: Metamorphic-Based Many-Objective Distillation of LLMs for Code-related Tasks

MORPH is a tool for **many-objective search-based knowledge distillation** of large language models (LLMs). It optimizes key trade-offs between model size, efficiency (Gigafactory FLOPS), accuracy, and robustness (measured via metamorphic testing). This repository includes:

1. **Source Code for MORPH**: The implementation of our many-objective optimization framework.
2. **Source Code for AVATAR**: A modified version of the baseline approach for model distillation with added functionality to calculate model robustness by measuring prediction flips on metamorphic inputs. The original AVATAR source code can be found at: https://github.com/soarsmu/Avatar

## Environment Setup

The artifact is designed to run in a Dockerized environment (https://www.docker.com).
To ensure reproducibility of experiments, we recommend using a machine equipped with GPUs and the NVIDIA CUDA toolkit. 
However, the tool can also run on CPUs, automatically falling back when no GPU is detected. 
If running on macOS (especially with Apple Silicon processors), check the compatibility of your Docker base image and dependencies.

### Prerequisites
- Docker (at least 8 GB of allocated memory).

### Notes on GPU vs. CPU Usage in Docker

The Dockerfile provided automatically installs the appropriate version of PyTorch based on the system’s CUDA compatibility. This means that whether you are running Docker on a server with GPUs or on a CPU-only system, the correct PyTorch version will be installed. However, there are considerations for both the build and run phases of the Docker container.

### Using Docker to Set Up the Environment

We provide a `Dockerfile` to simplify the environment setup process. Use the following commands to build and run the container:

1. **Build the Docker Image:**

   ```bash
   docker build -t morph_env .
   ```
   
**Note:** Ensure that the version of PyTorch specified in the `Dockerfile` matches your CUDA setup. For example, in our experiments, we used `torch==1.10.0+cu113`. Update this if your system uses a different CUDA version.

If your machine does not have a GPU (i.e., no CUDA support), 
you will need to use the CPU-compatible version of PyTorch instead. 
The Dockerfile automatically detects whether your machine has a GPU with CUDA support. 
If CUDA is available, it installs the GPU-compatible version of PyTorch. 
Otherwise, it installs the CPU-only version. You do not need to manually 
modify the Dockerfile to switch between GPU and CPU versions.

2. **Run the Docker Container:**

**For CUDA-enabled Machines (GPU):**
Use the --gpus all flag to enable GPU usage when running the container:

```bash
docker run -it -v YOUR_LOCAL_PATH:/root/Morph --gpus all morph_env
```

**For CPU-only Machines:**
On systems without GPU support (or if you want to run the container in CPU-only mode), omit the `--gpus all` flag:

```bash
docker run -it -v YOUR_LOCAL_PATH:/root/Morph morph_env
```

Replace `YOUR_LOCAL_PATH` with the directory where you have cloned this repository.

## Usage Instructions

To train and evaluate a model distilled by MORPH, use the script `train_and_evaluate_morph.py`:

```bash
python3 train_and_evaluate_morph.py --task <task> --model <model>
```
Replace `<task>` with `clone-detection` for code clone detection or	`vulnerability-detection` for vulnerability detection.
Replace `<model>` with `codebert` for CodeBERT or	`graphcodebert` for GraphCodeBERT.

For example, if you want to distill CodeBERT on the `clone-detection` task, run the command:
```bash
python3 train_and_evaluate_morph.py --task clone-detection --model codebert
```

**Outputs** MORPH will generate two results files:

* `/root/Morph/<model>/<task>/Morph/results_morph.csv` will contains the best configurations for the student/distilled model together with the performance metrics calculated on the test set, i.e, model size, accuracy, Giga FLOPS, and number of prediction flips.
* `/root/Morph/<model>/<task>/checkpoints/Morph/model.bin` is the trained distilled model


### Repository Breakdown:

1. **`CodeBERT`**:
   - **Clone-Detection**:
     - `Avatar`: Contains scripts and configurations for the baseline model.
     - `Morph`: Contains scripts and configurations for our MORPH tool.
     - `Data`: Includes the training, validation, and test sets, along with their metamorphic variants for robustness evaluation.
     - `checkpoint`: Stores the distilled models generated during the optimization process.
   - **Vulnerability-Detection**:
     - `Avatar`: Contains scripts and configurations for the baseline model.
     - `Morph`: Contains scripts and configurations for our MORPH tool.
     - `Data`: Includes the training, validation, and test sets, along with their metamorphic variants for robustness evaluation.
     - `checkpoint`: Stores the distilled models generated during the optimization process.

2. **`GraphCodeBERT`**:
   - **Clone-Detection**:
     - `Avatar`: Contains scripts and configurations for the baseline model.
     - `Morph`: Contains scripts and configurations for our MORPH tool.
     - `Data`: Includes the training, validation, and test sets, along with their metamorphic variants for robustness evaluation.
     - `checkpoint`: Stores the distilled models generated during the optimization process.
   - **Vulnerability-Detection**:
     - `Avatar`: Contains scripts and configurations for the baseline model.
     - `Morph`: Contains scripts and configurations for our MORPH tool.
     - `Data`: Includes the training, validation, and test sets, along with their metamorphic variants for robustness evaluation.
     - `checkpoint`: Stores the distilled models generated during the optimization process.


##Notes and Considerations

- **Variability in Results**: Neural network training and evolutionary algorithms (used for optimization) are non-deterministic. While the results may vary slightly, the overall trends and conclusions should remain consistent.
- **Compatibility on macOS:** For users running Docker on macOS, especially with Apple Silicon processors, ensure the compatibility of the Dockerfile base image and dependencies for proper execution.

