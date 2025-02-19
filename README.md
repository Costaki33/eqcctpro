# EQCCTPro

EQCCTPro is a seismic event detection and processing framework designed for efficient execution on CPU and GPU architectures. It leverages machine learning models to analyze mSEED seismic data files and optimize computational resources based on the system's configuration.

## Features
- Supports both CPU and GPU execution
- Configurable parallelism for optimized performance
- Automatic selection of best-use-case configurations
- Efficient handling of large-scale seismic data
- Includes tools for evaluating system performance

## Installation
To install the necessary dependencies, create a conda environment using:

```sh
conda env create -f environment.yml
conda activate eqcctpro
```

## Usage

### Running EQCCTPro
To run EQCCTPro with a specified mSEED input directory and output directory:

```python
from eqcctpro import EQCCTMSeedRunner

eqcct_runner = EQCCTMSeedRunner(
    use_gpu=True,
    intra_threads=1,
    inter_threads=1,
    cpu_id_list=[0,1,2,3,4],
    input_dir='/path/to/mseed',
    output_dir='/path/to/outputs',
    log_filepath='/path/to/outputs/eqcctpro.log',
    P_threshold=0.001,
    S_threshold=0.02,
    p_model_filepath='/path/to/model_p.h5',
    s_model_filepath='/path/to/model_s.h5',
    number_of_concurrent_predictions=5,
    best_usecase_config=True,
    csv_dir='/path/to/csv',
    selected_gpus=[0],
    set_vram_mb=24750,
    specific_stations='AT01, BP01, DG05'
)
eqcct_runner.run_eqcctpro()
```

### Evaluating System Performance
To evaluate the system’s GPU performance:

```python
from eqcctpro import EvaluateSystem

eval_gpu = EvaluateSystem(
    mode='gpu',
    intra_threads=1,
    inter_threads=1,
    input_dir='/path/to/mseed',
    output_dir='/path/to/outputs',
    log_filepath='/path/to/outputs/eqcctpro.log',
    csv_dir='/path/to/csv',
    P_threshold=0.001,
    S_threshold=0.02,
    p_model_filepath='/path/to/model_p.h5',
    s_model_filepath='/path/to/model_s.h5',
    stations2use=2,
    cpu_id_list=[0,1],
    set_vram_mb=24750,
    selected_gpus=[0]
)
eval_gpu.evaluate()
```

### Finding Optimal CPU/GPU Configurations
To determine the best CPU or GPU configuration:

```python
from eqcctpro import OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder

csv_filepath = '/path/to/csv'

cpu_finder = OptimalCPUConfigurationFinder(csv_filepath)
best_cpu_config = cpu_finder.find_best_overall_usecase()
print(best_cpu_config)

optimal_cpu_config = cpu_finder.find_optimal_for(cpu=3, station_count=2)
print(optimal_cpu_config)

gpu_finder = OptimalGPUConfigurationFinder(csv_filepath)
best_gpu_config = gpu_finder.find_best_overall_usecase()
print(best_gpu_config)

optimal_gpu_config = gpu_finder.find_optimal_for(num_cpus=1, gpu_list=[0], station_count=1)
print(optimal_gpu_config)
```

## Configuration
The `environment.yml` file specifies the dependencies required to run EQCCTPro. Ensure you have the correct versions installed by using the provided conda environment setup.

## License
EQCCTPro is provided under an open-source license. See LICENSE for details.

## Contact
For inquiries or issues, please contact [your email or organization link].

