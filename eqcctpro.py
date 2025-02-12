
import os 
import psutil 

from predictor import run_EQCCT_mseed, evaluate_system, find_optimal_configuration_cpu, find_optimal_configuration_gpu

input_mseed_directory_path = '/home/skevofilaxc/eqcctpro/mseed/20241215T115800Z_20241215T120100Z'   
output_pick_directory_path = '/home/skevofilaxc/eqcctpro/outputs'
log_file_path = '/home/skevofilaxc/eqcctpro/outputs/eqcctpro.log'
csv_filepath = '/home/skevofilaxc/eqcctpro/csv'

# Can run EQCCT on a given input dir on GPU or CPU 
# Can also specify the number of stations you want to use as well  
# run_EQCCT_mseed(use_gpu=True, 
#                 intra_threads=1, 
#                 inter_threads=1, 
#                 ray_cpus=5,
#                 input_dir=input_mseed_directory_path, 
#                 output_dir=output_pick_directory_path, 
#                 log_filepath=log_file_path, 
#                 P_threshold=0.001, 
#                 S_threshold=0.02, 
#                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
#                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', 
#                 number_of_concurrent_predictions=5,
#                 csv_dir=csv_filepath)

# Can evalue your system by using the test input directory
# Can either evaluate your system on the entire input dir or on a specific number of stations you want to evaluate 
# evaluate_system('gpu',
#                 intra_threads=1,
#                 inter_threads=1,
#                 input_dir=input_mseed_directory_path, 
#                 output_dir=output_pick_directory_path, 
#                 log_filepath=log_file_path,
#                 csv_dir=csv_filepath,
#                 P_threshold=0.001, 
#                 S_threshold=0.02, 
#                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
#                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',
#                 stations2use=2)

# Will return back the optimal amount of cpus and conc. pred. to use after doing the evaluation system function 
# cpus_to_use, num_concurrent_predictions, intra, inter, station_count = find_optimal_configuration_cpu(False, '/home/skevofilaxc/eqcctpro/csv/', 4, 2)

# Will return back the optimal amount of cpus, gpus, vram, and conc. conc. pred. to use after doing the evaluation system function 
cpus_to_use, num_concurrent_predictions, intra, inter, gpus, vram, station_count = find_optimal_configuration_gpu(False, '/home/skevofilaxc/eqcctpro/csv', 5, [0], 2)
