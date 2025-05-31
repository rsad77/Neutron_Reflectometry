import numpy as np
from scipy.signal import savgol_filter


def smooth_data(y, window_size=5, polynomial_order=2, method='savgol'):
    """
    Smooth data using specified method

    Args:
        y: Input data array
        window_size: Size of the smoothing window (must be odd)
        polynomial_order: Order of polynomial fit (for savgol)
        method: Smoothing method ('savgol' or 'moving_average')

    Returns:
        Smoothed data array
    """
    if len(y) < window_size:
        # For small datasets, reduce window size
        window_size = max(3, len(y) // 2)
        if window_size % 2 == 0:
            window_size -= 1

    if window_size % 2 == 0:
        window_size += 1  # Ensure window size is odd

    if method == 'savgol':
        try:
            return savgol_filter(y, window_size, polynomial_order)
        except ValueError:
            # Fallback to moving average if Savitzky-Golay fails
            return np.convolve(y, np.ones(window_size) / window_size, mode='same')
    elif method == 'moving_average':
        return np.convolve(y, np.ones(window_size) / window_size, mode='same')
    else:
        return y  # Return original if unknown method