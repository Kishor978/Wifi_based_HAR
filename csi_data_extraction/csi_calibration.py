import numpy as np

def csi_calibration(phase):
    """CSI phase calibration based on:
    https://github.com/ermongroup/Wifi_Activity_Recognition/blob/master/datfile_convert_final/phase_calibration.m
    
    """
    phases=np.array(phases)
    difference=0
    calibrated_phase,calibrated_phase_final=np.zeros_like(phase),np.zeros_like(phases)
    calibrated_phase[0] = phases[0]

    phases_len = phases.shape[0]

    for i in range(1, phases_len):
        temp = phases[i] - phases[i - 1]

        if abs(temp) > np.pi:
            difference = difference + 1 * np.sign(temp)

        calibrated_phase[i] = phases[i] - difference * 2 * np.pi

    k = (calibrated_phase[-1] - calibrated_phase[0]) / (phases_len - 1)
    b = np.mean(calibrated_phase)

    for i in range(phases_len):
        calibrated_phase_final[i] = calibrated_phase[i] - k * i - b

    return calibrated_phase_final


def calibrate_amplitude(amplitudes, rssi):  # Basic statistical normalization
    amplitudes = np.array(amplitudes)
    return ((amplitudes - np.min(amplitudes)) / (np.max(amplitudes) - np.min(amplitudes))) * rssi
