import os
import sys
import ray
import csv
import time
import glob
import obspy
import shutil
import random
import logging
import platform
import numpy as np
import pandas as pd
from os import listdir
from io import StringIO
from progress.bar import IncrementalBar
from contextlib import redirect_stdout
from datetime import datetime, timedelta
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import pynvml 
import warnings
import multiprocessing
warnings.filterwarnings("ignore")
from silence_tensorflow import silence_tensorflow
silence_tensorflow()


def recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall


def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

 
def f1(y_true, y_pred):
    precisionx = precision(y_true, y_pred)
    recallx = recall(y_true, y_pred)
    return 2*((precisionx*recallx)/(precisionx+recallx+tf.keras.backend.epsilon()))


def wbceEdit( y_true, y_pred) :
    ms = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_pred)) 
    ssim = 1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,1.0))
    return (ssim + ms)

w1 = 6000
w2 = 3
drop_rate = 0.2
stochastic_depth_rate = 0.1

positional_emb = False
conv_layers = 4
num_classes = 1
input_shape = (w1, w2)
num_classes = 1
input_shape = (6000, 3)
image_size = 6000  # We'll resize input images to this size
patch_size = 40  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size)
projection_dim = 40

num_heads = 4
transformer_units = [
    projection_dim,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = tf.keras.layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded
    
# Referred from: github.com:rwightman/pytorch-image-models.
class StochasticDepth(tf.keras.layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super(StochasticDepth, self).__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (len(tf.shape(x)) - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x
    


def convF1(inpt, D1, fil_ord, Dr):

    channel_axis = 1 if tf.keras.backend.image_data_format() == "channels_first" else -1
    #filters = inpt._keras_shape[channel_axis]
    filters = int(inpt.shape[-1])
    
    #infx = tf.keras.layers.Activation(tf.nn.gelu')(inpt)
    pre = tf.keras.layers.Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inpt)
    pre = tf.keras.layers.BatchNormalization()(pre)    
    pre = tf.keras.layers.Activation(tf.nn.gelu)(pre)
    
    #shared_conv = tf.keras.layers.Conv1D(D1,  fil_ord, strides =(1), padding='same')
    
    inf = tf.keras.layers.Conv1D(filters,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(pre)
    inf = tf.keras.layers.BatchNormalization()(inf)    
    inf = tf.keras.layers.Activation(tf.nn.gelu)(inf)
    inf = tf.keras.layers.Add()([inf,inpt])
    
    inf1 = tf.keras.layers.Conv1D(D1,  fil_ord, strides =(1), padding='same',kernel_initializer='he_normal')(inf)
    inf1 = tf.keras.layers.BatchNormalization()(inf1)  
    inf1 = tf.keras.layers.Activation(tf.nn.gelu)(inf1)    
    encode = tf.keras.layers.Dropout(Dr)(inf1)

    return encode


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        #x = tf.keras.layers.Dense(units, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size' : self.patch_size, 
            
        })
        
        return config
        
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, 1, 1],
            strides=[1, self.patch_size, 1, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches' : self.num_patches, 
            'projection_dim' : projection_dim, 
            
        })
        
        return config
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        
        #print(patch,positions)
        #temp = self.position_embedding(positions)
        #temp = tf.reshape(temp,(1,int(temp.shape[0]),int(temp.shape[1])))
        #encoded = tf.keras.layers.Add()([self.projection(patch), temp])
        #print(temp,encoded)
        
        return encoded


def create_cct_modelP(inputs):

    inputs1 = convF1(inputs,  10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = tf.keras.layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        #encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        #attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def create_cct_modelS(inputs):

    inputs1 = convF1(inputs,  10, 11, 0.1)
    inputs1 = convF1(inputs1, 20, 11, 0.1)
    inputs1 = convF1(inputs1, 40, 11, 0.1)
    
    inputreshaped = tf.keras.layers.Reshape((6000,1,40))(inputs1)
    # Augment data.
    #augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputreshaped)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    #print('done')
        
    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        encoded_patches = convF1(encoded_patches, 40,11, 0.1)
        # Layer normalization 1.
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        attention_output = convF1(attention_output, 40,11, 0.1)
    

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    #print(representation)
    ''' 
    attention_weights = tf.nn.softmax(tf.keras.layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(
        attention_weights, representation, transpose_a=True
    )
    weighted_representation = tf.squeeze(weighted_representation, -2)

    return weighted_representation
    '''
    return representation


def tf_environ(gpu_id, gpu_memory_limit_mb=None, gpus_to_use=None, intra_threads=None, inter_threads=None):
    print(r"""                     _                   
  ___  __ _  ___ ___| |_ _ __  _ __ ___  
 / _ \/ _` |/ __/ __| __| '_ \| '__/ _ \ 
|  __/ (_| | (_| (__| |_| |_) | | | (_) |
 \___|\__, |\___\___|\__| .__/|_|  \___/ 
         |_|            |_|                                
    """)
    print(f"\n-----------------------------\nTensorflow and Ray Configuration...\n")
    tf.debugging.set_log_device_placement(True)
    
    if gpu_id != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(map(str, gpus_to_use)) # Sets CUDA_VISIBLE_DEVICES = to the selected GPUs
        print(f"[{datetime.now()}] GPU processing enabled. Using GPUs: {gpus_to_use}")
        gpus = tf.config.experimental.list_physical_devices('GPU') # Lists all available GPUs to Tensorflow 
        
        if gpus:
            try:
                for gpu in gpus:
                    if gpu_memory_limit_mb:
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit_mb)]
                        )
                print(f"[{datetime.now()}] GPU and system memory configuration enabled.")
            except RuntimeError as e:
                print(f"Error configuring TensorFlow: {e}")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print(f"[{datetime.now()}] No GPUs found, using CPU.")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print(f"[{datetime.now()}] GPU processing disabled, using CPU.")
    if intra_threads is not None:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        print(f"[{datetime.now()}] Intraparallelism thread successfully set")
    if inter_threads is not None:
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
        print(f"[{datetime.now()}] Interparallelism thread successfully set")
    print(f"[{datetime.now()}] Tensorflow successfully set up for model operations")


def load_eqcct_model(input_modelP, input_modelS, log_file="results/logs/model.log"):
    # print(f"[{datetime.now()}] Loading EQCCT model.")
    
    # with open(log_file, mode="w", buffering=1) as log:
    #     log.write(f"*** Loading the model ...\n")

    # Model CCT
    inputs = tf.keras.layers.Input(shape=input_shape,name='input')

    featuresP = create_cct_modelP(inputs)
    featuresP = tf.keras.layers.Reshape((6000,1))(featuresP)

    featuresS = create_cct_modelS(inputs)
    featuresS = tf.keras.layers.Reshape((6000,1))(featuresS)

    logitp  = tf.keras.layers.Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_P')(featuresP)
    logits  = tf.keras.layers.Conv1D(1,  15, strides =(1), padding='same',activation='sigmoid', kernel_initializer='he_normal',name='picker_S')(featuresS)

    modelP = tf.keras.models.Model(inputs=[inputs], outputs=[logitp])
    modelS = tf.keras.models.Model(inputs=[inputs], outputs=[logits])

    model = tf.keras.models.Model(inputs=[inputs], outputs=[logitp,logits])

    summary_output = StringIO()
    with redirect_stdout(summary_output):
        model.summary()
    # log.write(summary_output.getvalue())
    # log.write('\n')

    sgd = tf.keras.optimizers.Adam()
    model.compile(optimizer=sgd,
                loss=['binary_crossentropy','binary_crossentropy'],
                metrics=['acc',f1,precision, recall])    
    
    modelP.load_weights(input_modelP)
    modelS.load_weights(input_modelS)

    # log.write(f"*** Loading is complete!")

    return model


def get_vram():
    """Prompt the user for VRAM input and ensure it's a valid float."""
    while True:
        try:
            vram = float(input("Enter how much VRAM the GPU can use (MB): ").strip())
            if vram > 0:
                return vram
            print("VRAM must be a positive number.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_gpu_vram():
    """Retrieve total and free VRAM (in GB) for the current GPU."""
    pynvml.nvmlInit()  # Initialize NVML
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Get first GPU
    total_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3)  # Convert bytes to GB
    free_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).free / (1024**3)  # Convert bytes to GB
    pynvml.nvmlShutdown()  # Shutdown NVML
    return total_vram, free_vram

def list_gpu_ids():
    """List all available GPU IDs on the system."""
    pynvml.nvmlInit()  # Initialize NVML
    gpu_count = pynvml.nvmlDeviceGetCount()  # Get number of GPUs
    gpu_ids = list(range(gpu_count))  # Create a list of GPU indices
    pynvml.nvmlShutdown()  # Shutdown NVML
    return gpu_ids

def get_valid_gpu_choice(available_gpus):
    """Prompt user to select GPU IDs and ensure they are valid."""
    while True:
        choice = input(f"Which GPU ID(s) would you like to use? (comma-separated, e.g., 0,1) or 'all' for all: ").strip()
        
        if choice.lower() == "all":
            return available_gpus  # Use all available GPUs
        
        try:
            selected_gpus = [int(x.strip()) for x in choice.split(",")]
            
            # Ensure all selected GPUs exist in the available list
            if all(gpu in available_gpus for gpu in selected_gpus):
                return selected_gpus
            else:
                print(f"Invalid choice. Please select from available GPUs: {available_gpus}")

        except ValueError:
            print("Invalid input. Please enter numeric GPU IDs separated by commas.")
            
            
            
            
def prepare_csv(csv_file_path, gpu:bool=False):
    """
    Loads or initializes the CSV file for storing test results.
    """
    if os.path.exists(csv_file_path):
        print(f"Loading existing CSV file from '{csv_file_path}'...")
        return pd.read_csv(csv_file_path)

    print(f"CSV file not found. Creating a new CSV file at '{csv_file_path}'...")
    columns = [
        "Trial Number", "Stations Used", "Number of Stations Used",
        "Number of CPUs Allocated for Ray to Use", "Number of Stations Running Predictions Concurrently",
        "Intra-parallelism Threads", "Inter-parallelism Threads", "Total Run time for Picker (s)",
        "Trial Success", "Error Message"
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_file_path, index=False)

def update_csv(csv_filepath, trial_number, intra, inter, success, error_message, output_dir):
    df = pd.read_csv(csv_filepath)
    df.at[trial_number -1, "Trial Number"] = trial_number
    df.at[trial_number -1, "Intra-parallelism Threads"] = intra
    df.at[trial_number -1, "Inter-parallelism Threads"] = inter
    df.at[trial_number -1, "Trial Success"] = success
    df.at[trial_number -1, "Error Message"] = error_message
    df.to_csv(csv_filepath, index=False)
    remove_directory(output_dir)


def generate_station_list(num_stations_to_use):
    if num_stations_to_use <= 10:
        return list(range(1, num_stations_to_use + 1))
    
    # Numbers 1-10
    station_list = list(range(1, 11))
    
    # Multiples of 5 up to num_stations_to_use
    multiples_of_5 = list(range(15, num_stations_to_use + 1, 5))
    
    # Any additional numbers between 21 and num_stations_to_use
    additional_numbers = list(range(21, num_stations_to_use + 1))
    
    # Combine lists while ensuring uniqueness
    return sorted(set(station_list + multiples_of_5 + additional_numbers))


def remove_directory(path):
    """
    Removes the specified directory if it exists.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Removed directory: {path}")
    else:
        print(f"Directory '{path}' does not exist anymore.")
        
        
def run_prediction(input_dir, output_dir, log_filepath, P_threshold, S_threshold, 
                   p_model_filepath, s_model_filepath, num_concurrent_predictions, 
                   ray_cpus, mode, use_gpu, stations2use, cpus_to_use,csv_filepath, intra_threads, inter_threads):
    """Function to run tf_environ and mseed_predictor as a separate process"""
    
    # Set CPU affinity for the child process
    pid = os.getpid()
    os.sched_setaffinity(pid, cpus_to_use)

    # Initialize TensorFlow environment
    tf_environ(gpu_id=-1, intra_threads=intra_threads, inter_threads=inter_threads)

    # Run prediction
    mseed_predictor(
        input_dir=input_dir, 
        output_dir=output_dir, 
        log_file=log_filepath, 
        P_threshold=P_threshold, 
        S_threshold=S_threshold, 
        p_model=p_model_filepath, 
        s_model=s_model_filepath, 
        number_of_concurrent_predictions=num_concurrent_predictions, 
        ray_cpus=ray_cpus,
        mode='network',
        use_gpu=use_gpu,
        stations2use=stations2use,
        testing=True,
        test_csv_filepath=csv_filepath
    )
 
 
def find_optimal_configurations_cpu(df):
    """
    Find:
    1. The best number of concurrent predictions for each (stations, CPUs) pair that results in the fastest runtime.
    2. The overall best configuration balancing stations, CPUs, and runtime.
    """

    # Convert relevant columns to numeric, handling NaNs gracefully
    df["Number of Stations Used"] = pd.to_numeric(df["Number of Stations Used"], errors="coerce")
    df["Number of CPUs Allocated for Ray to Use"] = pd.to_numeric(df["Number of CPUs Allocated for Ray to Use"], errors="coerce")
    df["Number of Stations Running Predictions Concurrently"] = pd.to_numeric(df["Number of Stations Running Predictions Concurrently"], errors="coerce")
    df["Total Run time for Picker (s)"] = pd.to_numeric(df["Total Run time for Picker (s)"], errors="coerce")

    # Drop rows with missing values in these essential columns
    df_cleaned = df.dropna(subset=["Number of Stations Used", "Number of CPUs Allocated for Ray to Use", 
                                "Number of Stations Running Predictions Concurrently", "Total Run time for Picker (s)"])

    # Find the best concurrent prediction configuration for each combination of (Stations, CPUs)
    optimal_concurrent_preds = df_cleaned.loc[
        df_cleaned.groupby(["Number of Stations Used", "Number of CPUs Allocated for Ray to Use"])
        ["Total Run time for Picker (s)"].idxmin()
    ]

    # Define what "moderate" means in terms of CPU usage (e.g., middle 50% of available CPUs)
    cpu_min = df_cleaned["Number of CPUs Allocated for Ray to Use"].quantile(0.25)
    cpu_max = df_cleaned["Number of CPUs Allocated for Ray to Use"].quantile(0.75)

    # Filter for rows within the moderate CPU range
    df_moderate_cpus = df_cleaned[(df_cleaned["Number of CPUs Allocated for Ray to Use"] >= cpu_min) & 
                                (df_cleaned["Number of CPUs Allocated for Ray to Use"] <= cpu_max)]

    # Sort by the highest number of stations first, then by the fastest runtime
    best_overall_config = df_moderate_cpus.sort_values(
        by=["Number of Stations Used", "Total Run time for Picker (s)"], 
        ascending=[False, True]  # Maximize stations, minimize runtime
    ).iloc[0]


    return optimal_concurrent_preds, best_overall_config


def find_optimal_configuration_cpu(cpu, station_count, best_overall_usecase: bool, eval_sys_results_dir:str): 
    # Check if eval_sys_results_dir is valid
    if not eval_sys_results_dir or not os.path.isdir(eval_sys_results_dir):
        print(f"Error: The provided directory path '{eval_sys_results_dir}' is invalid or does not exist.")
        print("Please provide a valid directory path for the input parameter 'csv_dir'.")
        return exit()  # Exit early if the directory is invalid
    
    if best_overall_usecase is True: 
        file_path = f"{eval_sys_results_dir}/best_overall_usecase.csv"

        # Check if the CSV file exists before reading
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' does not exist. Ensure the file is in the correct directory.")
            return exit()

        # Load the CSV
        df_best_overall = pd.read_csv(file_path)
        # Convert into a dictionary for easy access
        best_config_dict = df_best_overall.set_index(df_best_overall.columns[0]).to_dict()[df_best_overall.columns[1]]

        # Extract required values
        num_cpus = best_config_dict.get("Number of CPUs Allocated for Ray to Use")
        num_concurrent_predictions = best_config_dict.get("Number of Stations Running Predictions Concurrently")
        intra_threads = best_config_dict.get("Intra-parallelism Threads")
        inter_threads = best_config_dict.get("Inter-parallelism Threads")
        num_stations = best_config_dict.get("Number of Stations Used")
        total_runtime = best_config_dict.get("Total Run time for Picker (s)")
        
        print(f"CPU: {num_cpus}\n"
        f"Concurrent Predictions: {num_concurrent_predictions}\n"
        f"Intra-parallelism Threads: {intra_threads}\n"
        f"Inter-parallelism Threads: {inter_threads}\n"
        f"Stations: {num_stations}\n"
        f"Total Runtime (s): {total_runtime}")

        # Return the extracted values
        return int(float(num_cpus)), int(float(num_concurrent_predictions)), int(float(intra_threads)), int(float(inter_threads)), int(float(num_stations))
    
    else: # Optimal Configuration for User-Specified CPUs and Number of Stations to use
        file_path = f"{eval_sys_results_dir}/optimal_configurations.csv"

        # Check if the CSV file exists before reading
        if not os.path.exists(file_path):
            print(f"Error: The file '{file_path}' does not exist. Ensure the file is in the correct directory.")
            return exit() 
        
        df_optimal = pd.read_csv(file_path)

        # Convert relevant columns to numeric, handling NaNs gracefully
        df_optimal["Number of Stations Used"] = pd.to_numeric(df_optimal["Number of Stations Used"], errors="coerce")
        df_optimal["Number of CPUs Allocated for Ray to Use"] = pd.to_numeric(df_optimal["Number of CPUs Allocated for Ray to Use"], errors="coerce")
        df_optimal["Number of Stations Running Predictions Concurrently"] = pd.to_numeric(df_optimal["Number of Stations Running Predictions Concurrently"], errors="coerce")
        df_optimal["Total Run time for Picker (s)"] = pd.to_numeric(df_optimal["Total Run time for Picker (s)"], errors="coerce")
        filtered_df = df_optimal[
        (df_optimal["Number of CPUs Allocated for Ray to Use"] == cpu) &
        (df_optimal["Number of Stations Used"] == station_count)]
        if filtered_df.empty:
            print("No matching configuration found. Please enter a valid entry.")
            exit() 

        # Find the best configuration (fastest runtime)
        best_config = filtered_df.nsmallest(1, "Total Run time for Picker (s)").iloc[0]
        print(f"CPU: {cpu}\nConcurrent Predictions: {best_config['Number of Stations Running Predictions Concurrently']}\n"
            f"Intra-parallelism Threads: {best_config['Intra-parallelism Threads']}\n"
            f"Inter-parallelism Threads: {best_config['Inter-parallelism Threads']}\n"
            f"Stations: {station_count}\nTotal Runtime (s): {best_config['Total Run time for Picker (s)']}")

        return int(float(cpu)), int(float(best_config["Number of Stations Running Predictions Concurrently"])), int(float(best_config["Intra-parallelism Threads"])), int(float(best_config["Inter-parallelism Threads"])), int(float(station_count))



def evaluate_system(eval_mode:str, intra_threads:int, inter_threads:int, input_dir:str, output_dir:str, log_filepath, csv_dir, P_threshold, S_threshold, p_model_filepath, s_model_filepath, stations2use:int=None): 
    """
    evaluate_system will evaluate a given system's hardware to find the optimal amount of CPUs/GPUs and concurrent predictions to use 
    """
    # Set options to display all rows and columns
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 0)
    pd.set_option('display.max_colwidth', None)
    valid_modes = {"cpu", "gpu"}
    if eval_mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Choose either 'cpu' or 'gpu'.")
    
    csv_filepath = f"{csv_dir}/optimal_cpu_usage.csv" # Define csv filepath for test results 
    remove_directory(output_dir) # Remove output dir before it begins for maximum cleaning

    if stations2use is None: 
        stations2use_list = list(range(1, 11)) + list(range(15, 101, 5))
    else: 
        stations2use_list = generate_station_list(stations2use)
        
    if eval_mode == "cpu": 
        while True: 
            choice = input("Would you like to test your entire system (up to all CPUs available)? (y/n): ").strip().lower()
            if choice in {"y", "n"}:
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        
        if choice == "y": 
            cpu_count = os.cpu_count()
            cpu_list = list(range(cpu_count))
            print(f"[{datetime.now()}] Testing using up to all {cpu_count} available CPUs...")
                        
            prepare_csv(csv_filepath, False)
            trial_num = 1
            for i in range(1, cpu_count):
                cpus_to_use = cpu_list[:i]
                for j in stations2use_list:
                    # Define num of concurrent predictions per iteration
                    concurrent_predictions_list = generate_station_list(j) 
                    for k in concurrent_predictions_list: 
                        # Start a new process with CPU affinity
                        process = multiprocessing.Process(
                            target=run_prediction,
                            args=(input_dir, output_dir, log_filepath, P_threshold, 
                                  S_threshold, p_model_filepath, s_model_filepath, 
                                  k, i, eval_mode, False, j, cpus_to_use, csv_filepath, intra_threads, inter_threads)
                        )
                        process.start()
                        process.join()  # Wait for process to complete before continuing
                        
                        if process.exitcode == 0: 
                            update_csv(csv_filepath, trial_num, intra_threads, inter_threads, 1, "", output_dir)
                        else: 
                            update_csv(csv_filepath, trial_num, intra_threads, inter_threads, 0, process.exitcode, output_dir)
                        trial_num += 1
        if choice == "n": 
            cpu_count = int(input("How many CPUs would you like to use? (must be int): "))
            cpu_list = list(range(cpu_count))
            print(f"[{datetime.now()}] Testing using up to all {cpu_count} available CPUs...")
                        
            prepare_csv(csv_filepath, False)
            trial_num = 1
            for i in range(1, cpu_count+1):
                cpus_to_use = cpu_list[:i]
                for j in stations2use_list:
                    # Define num of concurrent predictions per iteration
                    concurrent_predictions_list = generate_station_list(j) 
                    for k in concurrent_predictions_list: 
                        # Start a new process with CPU affinity
                        process = multiprocessing.Process(
                            target=run_prediction,
                            args=(input_dir, output_dir, log_filepath, P_threshold, 
                                  S_threshold, p_model_filepath, s_model_filepath, 
                                  k, i, eval_mode, False, j, cpus_to_use, csv_filepath, intra_threads, inter_threads)
                        )
                        process.start()
                        process.join()  # Wait for process to complete before continuing
                        
                        if process.exitcode == 0: 
                            update_csv(csv_filepath, trial_num, intra_threads, inter_threads, 1, "", output_dir)
                        else: 
                            update_csv(csv_filepath, trial_num, intra_threads, inter_threads, 0, process.exitcode, output_dir)
                        trial_num += 1
            
        print(f"[{datetime.now()}] Testing complete.\n[{datetime.now()}] Finding Optimal Configurations...")
        # Compute optimal configurations (CPU)
        df = pd.read_csv(csv_filepath)
        optimal_configuration_df, best_overall_usecase_df = find_optimal_configurations_cpu(df)
        optimal_configuration_df.to_csv(f"{csv_dir}/optimal_configurations.csv")
        best_overall_usecase_df.to_csv(f"{csv_dir}/best_overall_usecase.csv")
        print(f"[{datetime.now()}] Optimal Configurations Found. Findings saved to:\n" 
                f" 1) Optimal CPU/Station/Concurrent Prediction Configurations: {csv_dir}/optimal_configurations.csv\n" 
                f" 2) Best Overall Usecase Configuration: {csv_dir}/best_overall_usecase.csv")
    
    if eval_mode == "gpu": 
        print(f"stations2use: {stations2use}")
        if stations2use is None: 
            stations2use_list = list(range(1, 11)) + list(range(15, 101, 5))
        else: 
            stations2use_list = generate_station_list(stations2use)
        
        while True: 
            choice = input("Would you like to set how much VRAM the GPU can use? (y/n): ").strip().lower()
            if choice in {"y", "n"}:
                break
            print(f"Invalid input. Please enter 'y' or 'n'.")
        
        if choice == "y": 
            free_vram_mb = get_vram()
            print(f"[{datetime.now()}] VRAM set to {vram} MB.")
        else:
            # Setting VRAM
            print(f"[{datetime.now()}] Utilizing available VRAM within Ray Memory Usage Threshold Limit of 0.95...")
            total_vram, available_vram = get_gpu_vram()
            print(f"[{datetime.now()}] Total VRAM: {total_vram:.2f} GB")
            print(f"[{datetime.now()}] Available VRAM: {available_vram:.2f} GB")
            # 95% of the Node's memory can be used by Ray and it's Raylets. 
            # Beyond the threshold, Ray will begin to kill process to save the node's memory             
            if available_vram / total_vram >= 0.9486: # 94.86% as a saftey value threshold, can use 94.85% and below 
                free_vram = total_vram * 0.9485        
            
            print(f"[{datetime.now()}] Using {round(free_vram, 2)} GB VRAM (within 94.85% VRAM threshold)")
            free_vram_mb = free_vram * 1024 # Convert to MB 
            
            # Setting GPUs to use 
            gpu_ids = list_gpu_ids()
            print(f"Available GPU IDs: {gpu_ids}")
            selected_gpus = get_valid_gpu_choice(gpu_ids)
            print(f"Using GPU(s): {selected_gpus}")
        
        
        prepare_csv(csv_filepath, True)
        trial_num = 1 
        

        for i in stations2use_list: 
            concurrent_predictions_list = generate_station_list(i)
            print(f"Concurrent predictions list: {concurrent_predictions_list}")
            for j in concurrent_predictions_list:    
                vram_per_task_mb = free_vram_mb / concurrent_predictions_list[j]
                # Define step size (5% of max)
                step_size = vram_per_task_mb * 0.05  
                # Generate range of values from 5% to 100% of vram_per_task_mb
                vram_steps = np.arange(step_size, vram_per_task_mb + step_size, step_size)
                for k in vram_steps: 
                    print(f"i in stations2uselist: {i}"
                          f"concurrent_predictions_list: {concurrent_predictions_list}"
                          f"concurrent_predictions_list[j]: {concurrent_predictions_list[j]}"
                          f"vram steps: {vram_steps}"
                          f"vram steps[k]: {vram_steps[k]}")
                     
                    
                    # # Start GPU process
                    # process = multiprocessing.Process(
                    #     target=run_gpu_prediction,
                    #     args=(input_dir, output_dir, log_filepath, P_threshold, 
                    #         S_threshold, p_model_filepath, s_model_filepath, 
                    #         k, i, eval_mode, True, j, gpus_to_use, csv_filepath, intra_threads, inter_threads)
                    # )
                    # process.start()
                    # process.join()  # Wait for process to complete

                    # # Handle exit codes
                    # if process.exitcode == 0:
                    #     update_csv(csv_filepath, trial_num, intra_threads, inter_threads, 1, "", output_dir)
                    # else:
                    #     update_csv(csv_filepath, trial_num, intra_threads, inter_threads, 0, process.exitcode, output_dir)
                    
        
        
        
        
        
        
        
        
        
        
        
        # tf_environ(gpu_id=1, gpu_memory_limit_mb=vram_per_task_mb, gpus_to_use=selected_gpus, intra_threads=intra_threads, inter_threads=inter_threads)
        # mseed_predictor(input_dir=input_dir, 
        #         output_dir=output_dir, 
        #         log_file=log_filepath, 
        #         P_threshold=P_threshold, 
        #         S_threshold=S_threshold, 
        #         p_model=p_model_filepath, 
        #         s_model=s_model_filepath, 
        #         number_of_concurrent_predictions=number_of_concurrent_predictions, 
        #         ray_cpus=ray_cpus,
        #         mode=mode,
        #         use_gpu=True,
        #         gpu_id=selected_gpus, 
        #         gpu_memory_limit_mb=vram_per_task_mb)
            
def run_EQCCT_mseed(
        use_gpu: bool, 
        ray_cpus: int, 
        input_dir: str, 
        output_dir: str, 
        log_filepath: str, 
        p_model_filepath: str, 
        s_model_filepath: str, 
        number_of_concurrent_predictions: int, 
        intra_threads: int = 1, 
        inter_threads: int = 1, 
        P_threshold: float = 0.001, 
        S_threshold: float = 0.02,
        specific_stations: str = None,
        csv_dir:str = None):
    """
    run_EQCCT_mseed enables users to use EQCCT to generate picks on MSEED files
    """
       
    # CPU Usage
    if use_gpu is False: 
        while True: 
            choice = input("Do you want to use best overall usecase configuration instead? (y/n): ").strip().lower()
            if choice in {"y", "n"}:
                break
            print("Invalid input. Please enter 'y' or 'n'.")
        
        if choice == "y":
            cpus_to_use, num_concurrent_predictions, intra, inter, station_count = find_optimal_configuration_cpu(1, 1, True, f'{csv_dir}')
            print(f"[{datetime.now()}] Using {cpus_to_use} CPUs, {num_concurrent_predictions} Conc. Predictions, {intra} Intra Threads, and {inter} Inter Threads")
            tf_environ(gpu_id=-1, intra_threads=intra, inter_threads=inter)
            mseed_predictor(input_dir=input_dir, 
                    output_dir=output_dir, 
                    log_file=log_filepath, 
                    P_threshold=P_threshold, 
                    S_threshold=S_threshold, 
                    p_model=p_model_filepath, 
                    s_model=s_model_filepath, 
                    number_of_concurrent_predictions=num_concurrent_predictions, 
                    ray_cpus=cpus_to_use,
                    use_gpu=False,
                    specific_stations=specific_stations)    
        else: 
            tf_environ(gpu_id=-1, intra_threads=intra_threads, inter_threads=inter_threads)
            mseed_predictor(input_dir=input_dir, 
                    output_dir=output_dir, 
                    log_file=log_filepath, 
                    P_threshold=P_threshold, 
                    S_threshold=S_threshold, 
                    p_model=p_model_filepath, 
                    s_model=s_model_filepath, 
                    number_of_concurrent_predictions=number_of_concurrent_predictions, 
                    ray_cpus=ray_cpus,
                    use_gpu=False,
                    specific_stations=specific_stations)
        
    # GPU Usage   
    if use_gpu is True: 
        while True: 
            choice = input("Would you like to set how much VRAM the GPU can use? (y/n): ").strip().lower()
            if choice in {"y", "n"}:
                break
            print(f"Invalid input. Please enter 'y' or 'n'.")
        
        if choice == "y": 
            free_vram_mb = get_vram()
            print(f"[{datetime.now()}] VRAM set to {vram} MB.")
        else:
            # Setting VRAM
            print(f"[{datetime.now()}] Utilizing available VRAM within Ray Memory Usage Threshold Limit of 0.95...")
            total_vram, available_vram = get_gpu_vram()
            print(f"[{datetime.now()}] Total VRAM: {total_vram:.2f} GB")
            print(f"[{datetime.now()}] Available VRAM: {available_vram:.2f} GB")
            # 95% of the Node's memory can be used by Ray and it's Raylets. 
            # Beyond the threshold, Ray will begin to kill process to save the node's memory             
            if available_vram / total_vram >= 0.9486: # 94.86% as a saftey value threshold, can use 94.85% and below 
                free_vram = total_vram * 0.9485        
            
            print(f"[{datetime.now()}] Using {round(free_vram, 2)} GB VRAM (within 94.85% VRAM threshold)")
            free_vram_mb = free_vram * 1024 # Convert to MB 
            
            # Setting GPUs to use 
            gpu_ids = list_gpu_ids()
            print(f"Available GPU IDs: {gpu_ids}")
            selected_gpus = get_valid_gpu_choice(gpu_ids)
            print(f"Using GPU(s): {selected_gpus}")
            
        vram_per_task_mb = free_vram_mb / number_of_concurrent_predictions
        
        tf_environ(gpu_id=1, gpu_memory_limit_mb=vram_per_task_mb, gpus_to_use=selected_gpus, intra_threads=intra_threads, inter_threads=inter_threads)
        mseed_predictor(input_dir=input_dir, 
                output_dir=output_dir, 
                log_file=log_filepath, 
                P_threshold=P_threshold, 
                S_threshold=S_threshold, 
                p_model=p_model_filepath, 
                s_model=s_model_filepath, 
                number_of_concurrent_predictions=number_of_concurrent_predictions, 
                ray_cpus=ray_cpus,
                use_gpu=True,
                gpu_id=selected_gpus, 
                gpu_memory_limit_mb=vram_per_task_mb)
        
    
    
    
        
def mseed_predictor(input_dir='downloads_mseeds',
              output_dir="detections",
              P_threshold=0.1,
              S_threshold=0.1, 
              normalization_mode='std',
              dt=1,
              batch_size=500,              
              overlap=0.3,
              gpu_id=None,
              gpu_limit=None,
              overwrite=False,
              log_file="./results/logs/picker/eqcct.log",
              stations2use=None,
              stations_filters=None,
              p_model=None,
              s_model=None,
              number_of_concurrent_predictions=None,
              ray_cpus=None,
              use_gpu=False,
              gpu_memory_limit_mb=None,
              testing=None,
              test_csv_filepath=None,
              specific_stations=None): 
    
    """ 
    
    To perform fast detection directly on mseed data.
    
    Parameters
    ----------
    input_dir: str
        Directory name containing hdf5 and csv files-preprocessed data.
            
    input_model: str
        Path to a trained model.
            
    stations_json: str
        Path to a JSON file containing station information. 
           
    output_dir: str
        Output directory that will be generated.
            
    P_threshold: float, default=0.1
        A value which the P probabilities above it will be considered as P arrival.                
            
    S_threshold: float, default=0.1
        A value which the S probabilities above it will be considered as S arrival.
            
    normalization_mode: str, default=std
        Mode of normalization for data preprocessing max maximum amplitude among three components std standard deviation.
             
    batch_size: int, default=500
        Batch size. This wont affect the speed much but can affect the performance. A value beteen 200 to 1000 is recommended.
             
    overlap: float, default=0.3
        If set the detection and picking are performed in overlapping windows.
             
    gpu_id: int
        Id of GPU used for the prediction. If using CPU set to None.        
             
    gpu_limit: float
       Set the maximum percentage of memory usage for the GPU. 

    overwrite: Bolean, default=False
        Overwrite your results automatically.
           
    Returns
    --------        
      
    """ 
    if use_gpu is False: 
        ray.init(ignore_reinit_error=True, num_cpus=ray_cpus, logging_level=logging.FATAL, log_to_driver=False) # Ray initalization using CPUs
        print(f"[{datetime.now()}] Ray Sucessfully Initialized with {ray_cpus} CPUs")
    elif use_gpu is True: 
        ray.init(ignore_reinit_error=True, num_gpus=len(gpu_id), num_cpus=ray_cpus, logging_level=logging.FATAL, log_to_driver=False) # Ray initalization using GPUs 
        print(f"[{datetime.now()}] Ray Sucessfully Initialized with {len(gpu_id)} GPU(s) and {ray_cpus} CPU(s)")
        
    args = {
    "input_dir": input_dir,
    "output_dir": output_dir,
    "P_threshold": P_threshold,
    "S_threshold": S_threshold,
    "normalization_mode": normalization_mode,
    "dt": dt,
    "overlap": overlap,
    "batch_size": batch_size,
    "overwrite": overwrite, 
    "gpu_id": gpu_id,
    "gpu_limit": gpu_limit,
    "p_model": p_model,
    "s_model": s_model,
    "stations_filters": stations_filters
    }

    # Ensure Output Dir exists 
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure logfile exists before continuing 
    if not os.path.exists(log_file): 
        print(f"Log file not found: '{log_file}'. Creating log file...")
        with open(log_file, "w") as f: 
            f.write("")
            print(f"Log file: {log_file} created.")
    else: 
        print(f"Log file '{log_file}' already exists.")
        
    with open(log_file, mode="w", buffering=1) as log:
        out_dir = os.path.join(os.getcwd(), str(args['output_dir']))    
           
        try:
            if platform.system() == 'Windows':
                station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("\\")[-1] != ".DS_Store"]
            else:     
                station_list = [ev.split(".")[0] for ev in listdir(args['input_dir']) if ev.split("/")[-1] != ".DS_Store"]

            station_list = sorted(set(station_list))
        except Exception as exp:
            log.write(f"{exp}\n")
            return
        log.write(f"[{datetime.now()}] GPU ID: {args['gpu_id']}; Batch size: {args['batch_size']}\n")
        log.write(f"[{datetime.now()}] {len(station_list)} station(s) in {args['input_dir']}\n")
        
        if stations2use and stations2use <= len(station_list):
            station_list = random.sample(station_list, stations2use)
            log.write(f"[{datetime.now()}] Using {len(station_list)} station(s) after selection.\n")
        if specific_stations is not None: 
            station_list = [x for x in station_list if x in specific_stations]
            log.write(f"[{datetime.now()}] Using {len(station_list)} station(s) after selection.\n")
    
        # print(f"station_list: {station_list}")
        tasks_predictor = [[f"({i+1}/{len(station_list)})", station_list[i], out_dir, args] for i in range(len(station_list))]
        # print(f"tasks_predictor:\n{tasks_predictor}")

        
        if not tasks_predictor:
            return

        # Submit tasks to ray in a queue
        tasks_queue = []
        max_pending_tasks = number_of_concurrent_predictions
        log.write(f"[{datetime.now()}] Started EQCCT picking process.\n")
        start_time = time.time() 
        print(f"\n-----------------------------\nEQCCT Pick Detection Process...\n\n[{datetime.now()}] Starting EQCCT...")
        print(f"[{datetime.now()}] Processing a total of {len(tasks_predictor)} stations, {max_pending_tasks} at a time...")
        with IncrementalBar("Processing Stations", max=len(tasks_predictor)) as bar:
            for i in range(len(tasks_predictor)):
                while True:
                    # Add new task to queue while max is not reached
                    if len(tasks_queue) < max_pending_tasks:
                        if use_gpu is False: 
                            tasks_queue.append(parallel_predict.remote(tasks_predictor[i], False, None))
                        elif use_gpu is True: 
                            gpu_allocation_per_task = len(gpu_id) / number_of_concurrent_predictions  # Ensure max_pending_tasks > 0 to avoid division by zero
                            task = parallel_predict.options(num_gpus=gpu_allocation_per_task, num_cpus=0).remote(tasks_predictor[i], True, gpu_memory_limit_mb)
                            tasks_queue.append(task)
                        break
                    # If there are more tasks than maximum, just process them
                    else:
                        tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                        for finished_task in tasks_finished:
                            log_entry = ray.get(finished_task)
                            log.write(log_entry + "\n")
                            log.flush()
                            bar.next()

            # After adding all the tasks to queue, process what's left
            while tasks_queue:
                tasks_finished, tasks_queue = ray.wait(tasks_queue, num_returns=1, timeout=None)
                for finished_task in tasks_finished:
                    log_entry = ray.get(finished_task)
                    log.write(log_entry + "\n")
                    log.flush()
                    bar.next() 
        log.write("------- END OF FILE -------\n")
        log.flush()
        end_time = time.time()
        print(f"[{datetime.now()}] EQCCT Pick Detection Process Complete! Picks are saved at {output_dir}\n[{datetime.now()}] Process Runtime: {end_time - start_time:.2f} s")

        
        if testing is True: 
            trial_data = {
                "Trial Number": None,
                "Stations Used": f"{station_list}",
                "Number of Stations Used": f"{len(station_list)}",
                "Number of CPUs Allocated for Ray to Use": f"{ray_cpus}",
                "Number of Stations Running Predictions Concurrently": f"{number_of_concurrent_predictions}",  
                "Intra-parallelism Threads": None, 
                "Inter-parallelism Threads": None, 
                "Total Run time for Picker (s)": f"{end_time - start_time:.6f}",
                "Trial Success": None, 
                "Error Message": None  
            }
            
            
            df_trial = pd.DataFrame([trial_data])
            
            # Check if the CSV file already exists
            if os.path.exists(test_csv_filepath):
                sys.stdout.flush() 
                # Load the existing CSV into a DataFrame
                df_existing = pd.read_csv(test_csv_filepath)
                # Append the trial data to the existing DataFrame
                df_existing = pd.concat([df_existing, df_trial], ignore_index=True)
            else:
                sys.stdout.flush()
            # Append the trial data directly to the CSV file
            df_trial.to_csv(test_csv_filepath, mode='a', index=False, header=not os.path.exists(test_csv_filepath))
            print(f"\nSuccessfully saved trial data to CSV at {test_csv_filepath}")

@ray.remote
def parallel_predict(predict_args, gpu=False, gpu_memory_limit_mb=None):
    
    if gpu is True: 
        # Ensure TensorFlow only sees its assigned VRAM
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_memory_limit_mb)]
                )
                print(f"[Task] VRAM Limited to {gpu_memory_limit_mb / 1024:.2f} GB")
            except RuntimeError as e:
                print(f"[Task] Error setting memory limit - {e}")
    
    pos, station, out_dir, args = predict_args
    model = load_eqcct_model(args["p_model"], args["s_model"])
    save_dir = os.path.join(out_dir, str(station)+'_outputs')
    csv_filename = os.path.join(save_dir,'X_prediction_results.csv')

    if os.path.isfile(csv_filename):
        if args['overwrite']:
            shutil.rmtree(save_dir)
        else:
            return f"[{datetime.now()}] {pos} {station}: Skipped (already exists - overwrite=False)."

    os.makedirs(save_dir)
    csvPr_gen = open(csv_filename, 'w')
    predict_writer = csv.writer(csvPr_gen, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    predict_writer.writerow(['file_name', 
                            'network',
                            'station',
                            'instrument_type',
                            'station_lat',
                            'station_lon',
                            'station_elv',
                            'p_arrival_time',
                            'p_probability',
                            's_arrival_time',
                            's_probability'])  
    csvPr_gen.flush()
    
    start_Predicting = time.time()
    # if mode == 'network': 
    files_list = glob.glob(f"{args['input_dir']}/{station}/*mseed")
    # if mode == 'single_station': 
    #     files_list = glob.glob(f"{args['input_dir']}/*mseed")
    
    try:
        meta, data_set, hp, lp = _mseed2nparray(args, files_list, station)
    except Exception: #InternalMSEEDError:
        return f"[{datetime.now()}] {pos} {station}: FAILED reading mSEED."

    try:
        params_pred = {'batch_size': args["batch_size"], 'norm_mode': args["normalization_mode"]}
        pred_generator = PreLoadGeneratorTest(meta["trace_start_time"], data_set, **params_pred)
        predP,predS = model.predict(pred_generator, verbose=0)
        
        detection_memory = []
        prob_memory=[]
        for ix in range(len(predP)):
            Ppicks, Pprob =  _picker(args, predP[ix,:, 0])   
            Spicks, Sprob =  _picker(args, predS[ix,:, 0], 'S_threshold')
            detection_memory,prob_memory=_output_writter_prediction(meta, csvPr_gen, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, ix,len(predP),len(predS))
                                        
        end_Predicting = time.time()
        delta = (end_Predicting - start_Predicting)
        return f"[{datetime.now()}] {pos} {station}: Finished the prediction in {round(delta,2)}s. (HP={hp}, LP={lp})"

    except Exception as exp:
        return f"[{datetime.now()}] {pos} {station}: FAILED the prediction. {exp}"


def _mseed2nparray(args, files_list, station):
    ' read miniseed files and from a list of string names and returns 3 dictionaries of numpy arrays, meta data, and time slice info'
          
    st = obspy.Stream()
    # Read and process files
    for file in files_list:
        temp_st = obspy.read(file)
        try:
            temp_st.merge(fill_value=0)
        except Exception:
            temp_st.merge(fill_value=0)
        temp_st.detrend('demean')
        if temp_st:
            st += temp_st
        else:
            return None  # No data to process, return early

    # Apply taper and bandpass filter
    max_percentage = 5 / (st[0].stats.delta * st[0].stats.npts) # 5s of data will be tapered
    st.taper(max_percentage=max_percentage, type='cosine')
    freqmin = 1.0
    freqmax = 45.0
    if args["stations_filters"] is not None:
        try:
            df_filters = args["stations_filters"]
            freqmin = df_filters[df_filters.sta == station].iloc[0]["hp"]
            freqmax = df_filters[df_filters.sta == station].iloc[0]["lp"]
        except:
            pass
    st.filter(type='bandpass', freqmin=freqmin, freqmax=freqmax, corners=2, zerophase=True)

    # Interpolate if necessary
    if any(tr.stats.sampling_rate != 100.0 for tr in st):
        try:
            st.interpolate(100, method="linear")
        except:
            st = _resampling(st)

    # Trim stream to the common start and end times
    st.trim(min(tr.stats.starttime for tr in st), max(tr.stats.endtime for tr in st), pad=True, fill_value=0)
    start_time = st[0].stats.starttime
    end_time = st[0].stats.endtime

    # Prepare metadata
    meta = {
        "start_time": start_time,
        "end_time": end_time,
        "trace_name": f"{files_list[0].split('/')[-2]}/{files_list[0].split('/')[-1]}"
    }
                
    # Prepare component mapping and types
    data_set = {}
    st_times = []
    components = {tr.stats.channel[-1]: tr for tr in st}
    time_shift = int(60 - (args['overlap'] * 60))

    # Define preferred components for each column
    components_list = [
        ['E', '1'],  # Column 0
        ['N', '2'],  # Column 1
        ['Z']        # Column 2
    ]

    current_time = start_time
    while current_time < end_time:
        window_end = current_time + 60
        st_times.append(str(current_time).replace('T', ' ').replace('Z', ''))
        npz_data = np.zeros((6000, 3))

        for col_idx, comp_options in enumerate(components_list):
            for comp in comp_options:
                if comp in components:
                    tr = components[comp].copy().slice(current_time, window_end)
                    data = tr.data[:6000]
                    # Pad with zeros if data is shorter than 6000 samples
                    if len(data) < 6000:
                        data = np.pad(data, (0, 6000 - len(data)), 'constant')
                    npz_data[:, col_idx] = data
                    break  # Stop after finding the first available component

        key = str(current_time).replace('T', ' ').replace('Z', '')
        data_set[key] = npz_data
        current_time += time_shift

    meta["trace_start_time"] = st_times

    # Metadata population with default placeholders for now
    try:
        meta.update({
            "receiver_code": st[0].stats.station,
            "instrument_type": 0,
            "network_code": 0,
            "receiver_latitude": 0,
            "receiver_longitude": 0,
            "receiver_elevation_m": 0
        })
    except Exception:
        meta.update({
            "receiver_code": station,
            "instrument_type": 0,
            "network_code": 0,
            "receiver_latitude": 0,
            "receiver_longitude": 0,
            "receiver_elevation_m": 0
        })
                    
    return meta, data_set, freqmin, freqmax


class PreLoadGeneratorTest(tf.keras.utils.Sequence):
    
    """ 
    
    Keras generator with preprocessing. For testing. Pre-load version.
    
    Parameters
    ----------
    list_IDsx: str
        List of trace names.
            
    file_name: str
        Path to the input hdf5 file.
            
    dim: tuple
        Dimension of input traces. 
           
    batch_size: int, default=32.
        Batch size.
            
    n_channels: int, default=3.
        Number of channels.
            
    norm_mode: str, default=max
        The mode of normalization, 'max' or 'std'                
            
    Returns
    --------        
    Batches of two dictionaries: {'input': X}: pre-processed waveform as input {'picker_P': y2, 'picker_S': y3}: outputs including two separate numpy arrays as labels for P, and S respectively.
    
    
    """

    def __init__(self, list_IDs, inp_data, batch_size=32, norm_mode='std', **kwargs):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.inp_data = inp_data        
        self.on_epoch_end()
        self.norm_mode = norm_mode
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        try:
            return int(np.ceil(len(self.list_IDs) / self.batch_size))
        except ZeroDivisionError:
            print("Your data duration in mseed file is too short! Try either longer files or reducing batch_size. ")

    def __getitem__(self, index):
        'Generate one batch of data'
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.list_IDs))
        indexes = self.indexes[start_idx:end_idx]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X = self.__data_generation(list_IDs_temp)

        # Handle case where the batch is not full
        if len(list_IDs_temp) < self.batch_size:
            X = X[:len(list_IDs_temp)]

        return {'input': X}

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
    
    def _normalize(self, data, mode='max'):
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            max_data[max_data == 0] = 1
            data /= max_data

        elif mode == 'std':
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
                       
    def __data_generation(self, list_IDs_temp):
        'readint the waveforms'
        X = np.zeros((self.batch_size, 6000, 3))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            data = self.inp_data[ID]
            data = self._normalize(data, self.norm_mode)
            X[i, :, :] = data
        return X


def _output_writter_prediction(meta, csvPr, Ppicks, Pprob, Spicks, Sprob, detection_memory,prob_memory,predict_writer, idx, cq, cqq):

    """ 
    
    Writes the detection & picking results into a CSV file.

    Parameters
    ----------
    dataset: hdf5 obj
        Dataset object of the trace.

    predict_writer: obj
        For writing out the detection/picking results in the CSV file. 
       
    csvPr: obj
        For writing out the detection/picking results in the CSV file.  

    matches: dic
        It contains the information for the detected and picked event.  
  
    snr: list of two floats
        Estimated signal to noise ratios for picked P and S phases.   
    
    detection_memory : list
        Keep the track of detected events.          
        
    Returns
    -------   
    detection_memory : list
        Keep the track of detected events.  
        
        
    """      

    station_name = meta["receiver_code"]
    station_lat = meta["receiver_latitude"]
    station_lon = meta["receiver_longitude"]
    station_elv = meta["receiver_elevation_m"]
    start_time = meta["trace_start_time"][idx]
    station_name = "{:<4}".format(station_name)
    network_name = meta["network_code"]
    network_name = "{:<2}".format(network_name)
    instrument_type = meta["instrument_type"]
    instrument_type = "{:<2}".format(instrument_type)  

    try:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S.%f')
    except Exception:
        start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        
    def _date_convertor(r):  
        if isinstance(r, str):
            mls = r.split('.')
            if len(mls) == 1:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S')
            else:
                new_t = datetime.strptime(r, '%Y-%m-%d %H:%M:%S.%f')
        else:
            new_t = r
            
        return new_t
  
    
    p_time = []
    p_prob = []
    PdateTime = []
    if Ppicks[0]!=None: 
#for iP in range(len(Ppicks)):
#if Ppicks[iP]!=None: 
        p_time.append(start_time+timedelta(seconds= Ppicks[0]/100))
        p_prob.append(Pprob[0])
        PdateTime.append(_date_convertor(start_time+timedelta(seconds= Ppicks[0]/100)))
        detection_memory.append(p_time) 
        prob_memory.append(p_prob)  
    else:          
        p_time.append(None)
        p_prob.append(None)
        PdateTime.append(None)

    s_time = []
    s_prob = []    
    SdateTime=[]
    if Spicks[0]!=None: 
#for iS in range(len(Spicks)):
#if Spicks[iS]!=None: 
        s_time.append(start_time+timedelta(seconds= Spicks[0]/100))
        s_prob.append(Sprob[0])
        SdateTime.append(_date_convertor(start_time+timedelta(seconds= Spicks[0]/100)))
    else:
        s_time.append(None)
        s_prob.append(None)
        SdateTime.append(None)

    SdateTime = np.array(SdateTime)
    s_prob = np.array(s_prob)
    
    p_prob = np.array(p_prob)
    PdateTime = np.array(PdateTime)
        
    predict_writer.writerow([meta["trace_name"], 
                     network_name,
                     station_name, 
                     instrument_type,
                     station_lat, 
                     station_lon,
                     station_elv,
                     PdateTime[0], 
                     p_prob[0],
                     SdateTime[0], 
                     s_prob[0]
                     ]) 



    csvPr.flush()                


    return detection_memory,prob_memory  


def _get_snr(data, pat, window=200):
    
    """ 
    
    Estimates SNR.
    
    Parameters
    ----------
    data : numpy array
        3 component data.    
        
    pat: positive integer
        Sample point where a specific phase arrives. 
        
    window: positive integer, default=200
        The length of the window for calculating the SNR (in the sample).         
        
    Returns
   --------   
    snr : {float, None}
       Estimated SNR in db. 
       
        
    """      
    import math
    snr = None
    if pat:
        try:
            if int(pat) >= window and (int(pat)+window) < len(data):
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)           
            elif int(pat) < window and (int(pat)+window) < len(data):
                window = int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)
            elif (int(pat)+window) > len(data):
                window = len(data)-int(pat)
                nw1 = data[int(pat)-window : int(pat)];
                sw1 = data[int(pat) : int(pat)+window];
                snr = round(10*math.log10((np.percentile(sw1,95)/np.percentile(nw1,95))**2), 1)    
        except Exception:
            pass
    return snr 


def _detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False):

    """
    
    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
        
    mph : {None, number}, default=None
        detect peaks that are greater than minimum peak height.
        
    mpd : int, default=1
        detect peaks that are at least separated by minimum peak distance (in number of data).
        
    threshold : int, default=0
        detect peaks (valleys) that are greater (smaller) than `threshold in relation to their immediate neighbors.
        
    edge : str, default=rising
        for a flat peak, keep only the rising edge ('rising'), only the falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        
    kpsh : bool, default=False
        keep peaks with same height even if they are closer than `mpd`.
        
    valley : bool, default=False
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Modified from 
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def _picker(args, yh3, thr_type='P_threshold'):
    """ 
    Performs detection and picking.

    Parameters
    ----------
    args : dic
        A dictionary containing all of the input parameters.  
        
    yh1 : 1D array
         probability. 

    Returns
    --------    
    Ppickall: Pick.
    Pproball: Pick Probability.                           
                
    """
    P_PICKall=[]
    Ppickall=[]
    Pproball = []
    perrorall=[]

    sP_arr = _detect_peaks(yh3, mph=args[thr_type], mpd=1)

    P_PICKS = []
    pick_errors = []
    if len(sP_arr) > 0:
        P_uncertainty = None  

        for pick in range(len(sP_arr)):        
            sauto = sP_arr[pick]


            if sauto: 
                P_prob = np.round(yh3[int(sauto)], 3) 
                P_PICKS.append([sauto,P_prob, P_uncertainty]) 

    so=[]
    si=[]
    P_PICKS = np.array(P_PICKS)
    P_PICKall.append(P_PICKS)
    for ij in P_PICKS:
        so.append(ij[1])
        si.append(ij[0])
    try:
        so = np.array(so)
        inds = np.argmax(so)
        swave = si[inds]
        Ppickall.append((swave))
        Pproball.append((np.max(so)))
    except:
        Ppickall.append(None)
        Pproball.append(None)

    #print(np.shape(Ppickall))
    #Ppickall = np.array(Ppickall)
    #Pproball = np.array(Pproball)
    
    return Ppickall, Pproball


def _resampling(st):
    'perform resampling on Obspy stream objects'
    
    need_resampling = [tr for tr in st if tr.stats.sampling_rate != 100.0]
    if len(need_resampling) > 0:
       # print('resampling ...', flush=True)    
        for indx, tr in enumerate(need_resampling):
            if tr.stats.delta < 0.01:
                tr.filter('lowpass',freq=45,zerophase=True)
            tr.resample(100)
            tr.stats.sampling_rate = 100
            tr.stats.delta = 0.01
            tr.data.dtype = 'int32'
            st.remove(tr)                    
            st.append(tr) 
    return st 


def _normalize(data, mode = 'max'):  
    """ 
    
    Normalize 3D arrays.
    
    Parameters
    ----------
    data : 3D numpy array
        3 component traces. 
        
    mode : str, default='std'
        Mode of normalization. 'max' or 'std'     
        
    Returns
    -------  
    data : 3D numpy array
        normalized data. 
            
    """  
       
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        assert(max_data.shape[-1] == data.shape[-1])
        max_data[max_data == 0] = 1
        data /= max_data              

    elif mode == 'std':               
        std_data = np.std(data, axis=0, keepdims=True)
        assert(std_data.shape[-1] == data.shape[-1])
        std_data[std_data == 0] = 1
        data /= std_data
    return data