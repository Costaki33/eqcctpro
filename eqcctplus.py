
import os 
from predictor import mseed_predictor, tf_environ

input_mseed_directory_path = '/home/skevofilaxc/eqcctplus/mseed/20241215T115800Z_20241215T120100Z'
output_pick_directory_path = '/home/skevofilaxc/eqcctplus/outputs'
log_file_path = '/home/skevofilaxc/eqcctplus/outputs/eqcctplus.log'

tf_environ(gpu_id=False, gpu_limit=None, intra_threads=1, inter_threads=1)
mseed_predictor(input_dir=input_mseed_directory_path, 
                output_dir=output_pick_directory_path, 
                log_file=log_file_path, 
                P_threshold=0.001, 
                S_threshold=0.02, 
                p_model='/home/skevofilaxc/model/ModelPS/test_trainer_024.h5', 
                s_model='/home/skevofilaxc/model/ModelPS/test_trainer_021.h5', 
                number_of_concurrent_predictions=5, ray_cpus=5)




