import numpy as np
import pandas as pd
import os

DATASET_FOLDER = "../dataset"
DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

SUBCARRIES_NUM_TWO_HHZ = 56
SUBCARRIES_NUM_FIVE_HHZ = 114

PHASE_MIN, PHASE_MAX = 3.1389, 3.1415
AMP_MIN, AMP_MAX = 0.0, 577.6582

def read_csi_data_from_csv(path_to_csv, is_five_hhz=False, antenna_pairs=4):
    """
    Read csi data(amplitude, phase) from .csv data

    :param path_to_csv: string
    :param is_five_hhz: boolean
    :param antenna_pairs: integer
    :return: (amplitudes, phases) => (np.array of shape(data len, num_of_subcarriers * antenna_pairs),
                                     np.array of shape(data len, num_of_subcarriers * antenna_pairs))
    """

    data = pd.read_csv(path_to_csv, header=None).values

    if is_five_hhz:
        subcarries_num = SUBCARRIES_NUM_FIVE_HHZ
    else:
        subcarries_num = SUBCARRIES_NUM_TWO_HHZ

    # 1 -> to skip subcarriers numbers in data
    amplitudes = data[:, subcarries_num * 1:subcarries_num * (1 + antenna_pairs)]
    phases = data[:, subcarries_num * (1 + antenna_pairs):subcarries_num * (1 + 2 * antenna_pairs)]

    return amplitudes, phases


def read_labels_from_csv(path_to_csv,expected_length):
    """
    Read labels (human activities) from csv file and remove unwanted classes.
    
    :param path_to_csv: string
    :return: filtered labels, np.array of shape (data_len, 1)
    """
    data = pd.read_csv(path_to_csv, header=None).values
    labels = data[:, 1]
    
    # Ensure labels match the expected length
    if len(labels) > expected_length:
        labels = labels[:expected_length]  # Trim extra labels if any
    elif len(labels) < expected_length:
        raise ValueError(f"Label file {path_to_csv} has fewer rows than CSI data!")


    # Filter out 'get_down' and 'get_up'
    valid_indices = ~np.isin(labels, ["get_down", "get_up"])
    
    return labels[valid_indices], valid_indices  # Also return indices to filter amplitudes & phases


def read_all_data_from_files(paths, is_five_hhz=True, antenna_pairs=4):
    """
    Read CSI and labels data from all folders in the dataset while removing 'get_down' and 'get_up'.

    :return: amplitudes, phases, labels all of shape (data len, num of subcarriers)
    """
    final_amplitudes, final_phases, final_labels = np.empty((0, antenna_pairs * SUBCARRIES_NUM_FIVE_HHZ)), \
                                                   np.empty((0, antenna_pairs * SUBCARRIES_NUM_FIVE_HHZ)), \
                                                   np.empty((0))

    for path in paths:
        amplitudes, phases = read_csi_data_from_csv(os.path.join(path, "data.csv"), is_five_hhz, antenna_pairs)
        labels, valid_indices = read_labels_from_csv(os.path.join(path, "label.csv"),len(amplitudes))

        # Apply the filter
        amplitudes, phases = amplitudes[valid_indices], phases[valid_indices]

        final_amplitudes = np.concatenate((final_amplitudes, amplitudes))
        final_phases = np.concatenate((final_phases, phases))
        final_labels = np.concatenate((final_labels, labels))

    return final_amplitudes, final_phases, final_labels


def read_all_data(is_five_hhz=True, antenna_pairs=4):
    all_paths = []

    for index, room in enumerate(DATA_ROOMS):
        for subroom in DATA_SUBROOMS[index]:
            all_paths.append(os.path.join(DATASET_FOLDER, room, subroom))

    return read_all_data_from_files(all_paths, is_five_hhz, antenna_pairs)