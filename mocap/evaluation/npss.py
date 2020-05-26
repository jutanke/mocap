import numpy as np

# ==============================================
# NPSS
# ==============================================
def NPSS(euler_gt_sequences, euler_pred_sequences):
    # computing 1) fourier coeffs 2)power of fft 3) normalizing power of fft dim-wise 4) cumsum over freq. 5) EMD
    gt_fourier_coeffs = np.zeros(euler_gt_sequences.shape, dtype=np.complex)
    pred_fourier_coeffs = np.zeros(euler_pred_sequences.shape, dtype=np.complex)

    # power vars
    gt_power = np.zeros((gt_fourier_coeffs.shape))
    pred_power = np.zeros((gt_fourier_coeffs.shape))

    # normalizing power vars
    gt_norm_power = np.zeros(gt_fourier_coeffs.shape)
    pred_norm_power = np.zeros(gt_fourier_coeffs.shape)

    cdf_gt_power = np.zeros(gt_norm_power.shape)
    cdf_pred_power = np.zeros(pred_norm_power.shape)

    emd = np.zeros(cdf_pred_power.shape[0:3:2])

    # used to store powers of feature_dims and sequences used for avg later
    seq_feature_power = np.zeros(euler_gt_sequences.shape[0:3:2])
    power_weighted_emd = 0

    for s in range(euler_gt_sequences.shape[0]):

        for d in range(euler_gt_sequences.shape[2]):
            gt_fourier_coeffs[s, :, d] = np.fft.fft(euler_gt_sequences[s, :, d])  # slice is 1D array
            pred_fourier_coeffs[s, :, d] = np.fft.fft(euler_pred_sequences[s, :, d])

            # computing power of fft per sequence per dim
            gt_power[s, :, d] = np.square(np.absolute(gt_fourier_coeffs[s, :, d]))
            pred_power[s, :, d] = np.square(np.absolute(pred_fourier_coeffs[s, :, d]))

            # matching power of gt and pred sequences
            gt_total_power = np.sum(gt_power[s, :, d])
            pred_total_power = np.sum(pred_power[s, :, d])
            # power_diff = gt_total_power - pred_total_power

            # adding power diff to zero freq of pred seq
            # pred_power[s,0,d] = pred_power[s,0,d] + power_diff

            # computing seq_power and feature_dims power
            seq_feature_power[s, d] = gt_total_power

            # normalizing power per sequence per dim
            if gt_total_power != 0:
                gt_norm_power[s, :, d] = gt_power[s, :, d] / gt_total_power

            if pred_total_power != 0:
                pred_norm_power[s, :, d] = pred_power[s, :, d] / pred_total_power

            # computing cumsum over freq
            cdf_gt_power[s, :, d] = np.cumsum(gt_norm_power[s, :, d])  # slice is 1D
            cdf_pred_power[s, :, d] = np.cumsum(pred_norm_power[s, :, d])

            # computing EMD
            emd[s, d] = np.linalg.norm((cdf_pred_power[s, :, d] - cdf_gt_power[s, :, d]), ord=1)

    # computing weighted emd (by sequence and feature powers)
    power_weighted_emd = np.average(emd, weights=seq_feature_power)

    return power_weighted_emd