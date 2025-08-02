import numpy as np
import pandas as pd
import torch


#Working With Chat-gpt sorrowfully generates a lot of low quality code. That needs helper methods.
#After that organizing that code in a way that makes sense becomes challenging


def apply_correction(standard_prediction, correction_ratio):
    corrected_prediction = standard_prediction * correction_ratio
    return corrected_prediction

def compute_correction_ratio(testing_data, testing_prediction, save=False, save_path="correction_ratios.csv"):
    data = np.array(testing_data)
    pred = np.array(testing_prediction)

    if data.shape != pred.shape:
        raise ValueError("Shape mismatch between testing_data and testing_prediction")

    if data.ndim == 2:
        if data.shape[1] == 1:
            data = data.flatten()
            pred = pred.flatten()
        elif data.shape[0] == data.shape[1]:
            data = np.diag(data)
            pred = np.diag(pred)
        else:
            raise ValueError(f"Ambiguous shape {data.shape}. Please reshape manually.")

    correction_ratio = data / pred

    if save:
        df = pd.DataFrame({'correction_ratio': correction_ratio})
        df.to_csv(save_path, index=False)
        print(f"Saved correction ratios to {save_path}")

    return correction_ratio  # Return as NumPy array

def flatten_predictions(preds):
    # Handles torch, numpy, or nested lists
    flat = []
    try:
        for p in preds:
            if hasattr(p, 'item'):
                flat.append(p.item())
            elif isinstance(p, (list, tuple, np.ndarray)):
                flat.append(p[0] if hasattr(p[0], 'item') else p[0])
            else:
                flat.append(p)
    except Exception as e:
        raise ValueError(f"Could not flatten predictions: {e}")
    return np.array(flat)

def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, 'numpy'):
        return arr.numpy()
    if isinstance(arr, (list, tuple)):
        return np.array(arr)
    raise ValueError("Unsupported data type for actual values.")

def move_on(batch, window, i):
    """
    In the i-th step:
    - Remove the first `min(i, len(window))` elements from the current window
    - Append `min(i, len(batch))` new elements from the batch
    Ensures the resulting window has the same fixed length.

    Handles edge cases when batch is longer than window or i > len(window).
    """
    window = torch.tensor(window, dtype=torch.float32)
    batch = torch.tensor(batch, dtype=torch.float32)

    if i == 0:
        return window

    # Determine how many to shift
    shift = min(i, len(window))
    insert = min(i, len(batch))

    # New part to append
    appended = batch[:insert]

    # Handle corner case if appended is longer than window
    if len(appended) >= len(window):
        return appended[-len(window):]  # Take last 'window size' values

    # Remove from window, append from batch
    new_window = torch.cat([window[shift:], appended])

    # If undefiled, pad front with zeros
    if len(new_window) < len(window):
        pad_len = len(window) - len(new_window)
        new_window = torch.cat([torch.zeros(pad_len), new_window])
    return new_window

def batch_window_generator(data, batch_size, window_size):
    list_batch_window_tuple = []
    total_samples = len(data)

    # Convert pandas DataFrame/Series to numpy array if needed
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()
    elif not isinstance(data, np.ndarray):
        raise TypeError(f"Unsupported data type: {type(data)}")

    for i in range(0, total_samples, batch_size):
        batch = data[i: i + batch_size]

        # Construct the window
        if i >= window_size:
            window = data[i - window_size: i]
        else:
            window = np.zeros(window_size)

        list_batch_window_tuple.append((batch, window))

    return list_batch_window_tuple

