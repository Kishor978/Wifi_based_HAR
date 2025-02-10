from torch.utils.data import Dataset, DataLoader
from sklearn import decomposition
import numpy as np


from utils import read_all_data_from_files, calibrate_amplitude, dwn_noise, hampel

DATASET_FOLDER = "../dataset"
DATA_ROOMS = ["bedroom_lviv", "parents_home", "vitalnia_lviv"]
DATA_SUBROOMS = [["1", "2", "3", "4"], ["1"], ["1", "2", "3", "4", "5"]]

SUBCARRIES_NUM_TWO_HHZ = 56
SUBCARRIES_NUM_FIVE_HHZ = 114

PHASE_MIN, PHASE_MAX = 3.1389, 3.1415
AMP_MIN, AMP_MAX = 0.0, 577.6582


def read_csi_data_from_csv(csis, is_five_hhz=True, antenna_pairs=4):
    """
    Read csi data(amplitude, phase) from .csv data

    :param path_to_csv: string
    :param is_five_hhz: boolean
    :param antenna_pairs: integer
    :return: (amplitudes, phases) => (np.array of shape(data len, num_of_subcarriers * antenna_pairs),
                                     np.array of shape(data len, num_of_subcarriers * antenna_pairs))
    """


    if is_five_hhz:
        subcarries_num = SUBCARRIES_NUM_FIVE_HHZ
    else:
        subcarries_num = SUBCARRIES_NUM_TWO_HHZ
    
    amplitudes_list = []
    phases_list = []

    for data in csis:
        if len(data) != subcarries_num * antenna_pairs * 2:
            raise ValueError(f"Data length mismatch: expected {subcarries_num * antenna_pairs * 2}, got {len(data)}")

        amplitudes = data[:subcarries_num * antenna_pairs]
        phases = data[subcarries_num * antenna_pairs:]
        
        amplitudes_list.append(amplitudes)
        phases_list.append(phases)

    return np.array(amplitudes_list), np.array(phases_list)
        



class CSIDataset(Dataset):
    """CSI Dataset."""

    def __init__(self, train_csi, labels, window_size=32, step=1,is_training=False):
        self.is_training = is_training
        self.amplitudes, self.phases= read_csi_data_from_csv(train_csi)
        print("len",len(self.amplitudes))
        print(self.phases.shape)
        self.labels = labels
 
        self.amplitudes = calibrate_amplitude(self.amplitudes)

        pca = decomposition.PCA(n_components=3)

        # self.phases[:, 0 * SUBCARRIES_NUM_FIVE_HHZ:1 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 0 * SUBCARRIES_NUM_FIVE_HHZ:1 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 1 * SUBCARRIES_NUM_FIVE_HHZ:2 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 1 * SUBCARRIES_NUM_FIVE_HHZ:2 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 2 * SUBCARRIES_NUM_FIVE_HHZ:3 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 2 * SUBCARRIES_NUM_FIVE_HHZ:3 * SUBCARRIES_NUM_FIVE_HHZ]))
        # self.phases[:, 3 * SUBCARRIES_NUM_FIVE_HHZ:4 * SUBCARRIES_NUM_FIVE_HHZ] = calibrate_amplitude(calibrate_phase(
        #     self.phases[:, 3 * SUBCARRIES_NUM_FIVE_HHZ:4 * SUBCARRIES_NUM_FIVE_HHZ]))

        self.amplitudes_pca = []

        data_len = self.phases.shape[0]
        print('data_len',data_len)
        for i in range(self.phases.shape[1]):
            self.amplitudes[:data_len, i] = dwn_noise(hampel(self.amplitudes[:, i]))[
                :data_len
            ]

        for i in range(4):
            self.amplitudes_pca.append(
                pca.fit_transform(
                    self.amplitudes[
                        :,
                        i * SUBCARRIES_NUM_FIVE_HHZ : (i + 1) * SUBCARRIES_NUM_FIVE_HHZ,
                    ]
                )
            )
        self.amplitudes_pca = np.array(self.amplitudes_pca)
        self.amplitudes_pca = self.amplitudes_pca.reshape(
            (
                self.amplitudes_pca.shape[1],
                self.amplitudes_pca.shape[0] * self.amplitudes_pca.shape[2],
            )
        )

        self.label_keys = list(set(self.labels))
        self.class_to_idx = {
            "standing": 0,
            "walking": 1,
            # "get_down": 2,
            "sitting": 2,
            # "get_up": 4,
            "lying": 3,
            "no_person": 4,
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        self.window = window_size
        if window_size == -1:
            self.window = self.labels.shape[0] - 1

        self.step = step

    def __getitem__(self, idx):
        if self.window == 0:
            return (
                np.append(self.amplitudes[idx], self.phases[idx]),
                self.class_to_idx[self.labels[idx + self.window - 1]],
            )

        idx = idx * self.step
        all_xs, all_ys = [], []

        for index in range(idx, idx + self.window):
        # Load the amplitude and PCA data for this timestep
            amplitude = self.amplitudes[index]
            pca = self.amplitudes_pca[index]

            # Combine features
            combined = np.append(amplitude, pca)

            # Add noise to this individual sample (not the entire window)
            if self.is_training:
                noise = np.random.normal(0, 0.01, size=combined.shape)
                combined += noise

            all_xs.append(combined)


        return np.array(all_xs), self.class_to_idx[self.labels[idx + self.window - 1]]

    def __len__(self):
        return int((self.labels.shape[0] - self.window) // self.step) + 1


if __name__ == "__main__":
    val_dataset = CSIDataset(
        [
            "./dataset/bedroom_lviv/4",
        ]
    )

    dl = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1)

    for i in dl:
        print(i[0].shape)
        print(i[1].shape)

        break

    print(val_dataset[0])
