
import os 
import psutil 

from predictor import run_EQCCT_mseed

input_mseed_directory_path = '/home/skevofilaxc/eqcctplus/mseed/20241215T115800Z_20241215T120100Z' # '/home/skevofilaxc/eqcctplus/ALPN'    
output_pick_directory_path = '/home/skevofilaxc/eqcctplus/outputs'
log_file_path = '/home/skevofilaxc/eqcctplus/outputs/eqcctplus.log'


# Get the first 5 CPU cores (0-4)
cpu_cores_to_use = list(range(5))  # [0, 1, 2, 3, 4]

# Set CPU affinity for the current process
psutil.Process(os.getpid()).cpu_affinity(cpu_cores_to_use)

print(f"Process limited to CPUs: {cpu_cores_to_use}")

run_EQCCT_mseed(use_gpu=False, 
                intra_threads=1, 
                inter_threads=1, 
                ray_cpus=5,
                mode='network', 
                input_dir=input_mseed_directory_path, 
                output_dir=output_pick_directory_path, 
                log_filepath=log_file_path, 
                P_threshold=0.001, 
                S_threshold=0.02, 
                p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
                s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', 
                number_of_concurrent_predictions=5)



