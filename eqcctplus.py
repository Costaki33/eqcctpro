
import os 
import psutil 

from predictor import run_EQCCT_mseed, evaluate_system, find_optimal_configuration_cpu

input_mseed_directory_path = '/home/skevofilaxc/eqcctplus/mseed/20241215T115800Z_20241215T120100Z' # '/home/skevofilaxc/eqcctplus/ALPN'    
output_pick_directory_path = '/home/skevofilaxc/eqcctplus/outputs'
log_file_path = '/home/skevofilaxc/eqcctplus/outputs/eqcctplus.log'
csv_filepath = '/home/skevofilaxc/eqcctplus/csv'

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
                number_of_concurrent_predictions=5,
                specific_stations = 'ALPN, VHRN, PB35',
                csv_dir = '/home/skevofilaxc/eqcctplus/csv')

# evaluate_system('cpu',
#                 stations2use=5,
#                 intra_threads=1,
#                 inter_threads=1,
#                 input_dir=input_mseed_directory_path, 
#                 output_dir=output_pick_directory_path, 
#                 log_filepath=log_file_path,
#                 csv_dir=csv_filepath,
#                 P_threshold=0.001, 
#                 S_threshold=0.02, 
#                 p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
#                 s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5',)

# cpus_to_use, num_concurrent_predictions, intra, inter, station_count = find_optimal_configuration_cpu(4, 3, True, '/home/skevofilaxc/eqcctplus/csv/')