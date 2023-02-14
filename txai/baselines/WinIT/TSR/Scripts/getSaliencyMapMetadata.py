import os
import pathlib

import numpy as np

from .Plotting.plot import plotExampleBox


def getSaliencyMapMetadata(saliency_dir, output_dir, specific_inputs=()):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for file in os.listdir(saliency_dir):
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.npy':
            cut_filename = filename[9:-9]
            saliency_data = np.load(saliency_dir + '/' + file)
            plotExampleBox(np.average(saliency_data, axis=0), output_dir + '/' + cut_filename + '_mean', greyScale=True, flip=True)
            plotExampleBox(np.std(saliency_data, axis=0), output_dir + '/' + cut_filename + '_std', greyScale=True, flip=True)
            for index in specific_inputs:
                plotExampleBox(saliency_data[index], output_dir + '/' + cut_filename + f'_{index}', greyScale=True, flip=True)


if __name__ == '__main__':
    SALIENCY_DIR = 'Results/Saliency_Values'
    OUTPUT_DIR = 'Graphs/Saliency_Maps/Average_Maps'
    SPECIFIC_INPUTS = [0, 10, 20, 30]
    getSaliencyMapMetadata(SALIENCY_DIR, OUTPUT_DIR, SPECIFIC_INPUTS)
