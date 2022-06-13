import os
import Augmentor

def augment(source_folder):
    for i in os.listdir(source_folder):
        if len(os.listdir(source_folder + "/"+i))==0:
            continue 
        p = Augmentor.Pipeline(source_folder + '/'+ i, output_directory=source_folder + "/"+i)

        # Defining augmentation parameters and generating 5 samples
        p.flip_left_right(0.5)
        # p.black_and_white(0.1)
        p.rotate(0.3, 10, 10)
        p.skew(0.4, 0.5)
        p.zoom(probability = 0.2, min_factor = 1.1, max_factor = 1.5)
        p.sample(100-len(os.listdir(source_folder + "/"+i)))