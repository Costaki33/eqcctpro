# from eqcctpro import EQCCTMSeedRunner, EvaluateSystem, OptimalCPUConfigurationFinder, OptimalGPUConfigurationFinder
import eqcctpro as eq
input_mseed_directory_path = '/home/skevofilaxc/eqcctpro/mseed/20241215T115800Z_20241215T120100Z'   
output_pick_directory_path = '/home/skevofilaxc/eqcctpro/outputs'
log_file_path = '/home/skevofilaxc/eqcctpro/outputs/eqcctpro.log'
csv_filepath = '/home/skevofilaxc/eqcctpro/csv'

# Can run EQCCT on a given input dir on GPU or CPU 
# Can also specify the number of stations you want to use as well  




eqcct_runner = eq.EQCCTMSeedRunner(use_gpu=True, 
                intra_threads=1, 
                inter_threads=1, 
                cpu_id_list=[0,1,2,3,4],
                input_dir=input_mseed_directory_path, 
                output_dir=output_pick_directory_path, 
                log_filepath=log_file_path, 
                P_threshold=0.001, 
                S_threshold=0.02, 
                p_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
                s_model_filepath='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', 
                number_of_concurrent_predictions=3,
                best_usecase_config=False,
                csv_dir=csv_filepath,
                selected_gpus=[0],
                set_vram_mb=24750,
                specific_stations='AT01, BP01, DG05')

eqcct_runner.run_eqcctpro()

# import tensorflow as tf

# print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# # Run a simple matrix multiplication on GPU
# with tf.device('/GPU:0'):
#     a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
#     b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
#     c = tf.matmul(a, b)
#     print("Matrix multiplication result:", c)
