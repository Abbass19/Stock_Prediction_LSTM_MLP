import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler, RobustScaler


#Function that loads data and splits it and preprocess it as well
def dataloader(csv_path = None, target_column: str = "MPN5P"):
    csv_path = os.path.join(os.path.dirname(__file__), "..", "Data", "my_data.csv")
    sheet = pd.read_csv(csv_path)
    data = sheet[target_column]
    train_data, test_data = split_data(data)
    train_data, test_data, scaler_1, scaler_2 = feature_1_normalize(train_data, test_data)
    return train_data, test_data, scaler_1, scaler_2


#Two important functions that preprocess the data. Solving it's skewness through logarithmic function.
# And also does scaling for better model training and convergence.
def feature_1_normalize(train_data, test_data):

    #Applying Log Transformer to Solve Skewness of Data
    train_data = np.log1p(train_data)
    test_data = np.log1p(test_data)
    #Applying Robust Scaler to Solve Outliers Reversibly
    scaler_1 = RobustScaler()
    train_data = np.array(train_data).reshape(-1, 1)
    train_data = scaler_1.fit_transform(train_data)
    test_data = np.array(test_data).reshape(-1, 1)
    test_data = scaler_1.transform(test_data)
    scaler_2 = MinMaxScaler()
    train_data = scaler_2.fit_transform(train_data)
    test_data = scaler_2.transform(test_data)
    return train_data.flatten(), test_data.flatten() , scaler_1, scaler_2

def feature_1_denormalize(scaled_data, scaler_1, scaler_2):

    scaled_data = np.array(scaled_data).reshape(-1, 1)
    data_scaled_1 = scaler_2.inverse_transform(scaled_data)
    # Reverse RobustScaler
    data_logged = scaler_1.inverse_transform(data_scaled_1)
    # Reverse log1p
    data_original = np.expm1(data_logged)

    return data_original.flatten()



# A very important function that analysis the kurtosis and skewness of data.
# revealing how ready is the data to be fitted to a model.
def analyze_distribution(data):
    print("📊 Data Distribution Analysis\n")

    # Basic Stats
    mean = np.mean(data)
    std = np.std(data)
    cv = std / mean if mean != 0 else float('inf')
    data_skew = skew(data)
    data_kurtosis = kurtosis(data)

    # IQR and Outliers
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    outlier_ratio = np.sum(outliers) / len(data)

    # Percentile Ratios
    p70 = np.percentile(data, 70)
    p99 = np.percentile(data, 99)
    p_ratio = p99 / p70 if p70 != 0 else float('inf')

    # Judgments
    judgments = {
        'CV':          ("✅ Good" if cv < 1.0 else "⚠️ High variability"),
        'Skewness':    ("✅ Symmetric" if abs(data_skew) < 0.5 else "⚠️ Skewed"),
        'Kurtosis':    ("✅ Normal-like" if -1 < data_kurtosis < 3 else "⚠️ Extreme tails"),
        'Outliers':    ("✅ Low outliers" if outlier_ratio < 0.05 else "⚠️ Too many outliers"),
        'P-Ratio':     ("✅ Stable spread" if p_ratio < 2.0 else "⚠️ Top-end dominance")
    }

    # Print Metrics & Judgments
    print(f"🔸 Mean: {mean:.4f}")
    print(f"🔸 Std Dev: {std:.4f}")
    print(f"🔸 Coefficient of Variation (CV): {cv:.4f} → {judgments['CV']}")
    print(f"🔸 Skewness: {data_skew:.4f} → {judgments['Skewness']}")
    print(f"🔸 Kurtosis: {data_kurtosis:.4f} → {judgments['Kurtosis']}")
    print(f"🔸 IQR: {IQR:.4f}")
    print(f"🔸 Outlier Ratio (Tukey's rule): {outlier_ratio * 100:.2f}% → {judgments['Outliers']}")
    print(f"🔸 99th / 70th Percentile: {p_ratio:.2f} → {judgments['P-Ratio']}")
    print()

    # Overall Readiness
    readiness = all("✅" in v for v in judgments.values())
    if readiness:
        print("✅ The data appears ready for model training.\n")
    else:
        print("⚠️ Data may require preprocessing (transformation or cleaning) before training.\n")

    # Plots
    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data, bins=40, edgecolor='black', alpha=0.7)
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # CDF
    plt.subplot(1, 2, 2)
    sorted_data = np.sort(data)
    cdf = np.arange(len(data)) / len(data)
    plt.plot(sorted_data, cdf)
    plt.title('CDF (Cumulative Distribution Function)')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


#A very important function used in training to create set of training
# and validation sets of data.
def rolling_forecast_origin_split(data, n_splits=5, val_size=20):

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data = data.to_numpy()

    total_len = len(data)
    train_val_pairs = []

    for i in range(n_splits):
        train_end = total_len - val_size * (n_splits - i)
        val_start = train_end
        val_end = val_start + val_size

        train = data[:train_end]
        val = data[val_start:val_end]

        train_val_pairs.append((train, val))

    return train_val_pairs

# A function used in some cases. A helper method for
def get_dates_from_csv(date_column: str = "DCP"):
    csv_path = os.path.join(os.path.dirname(__file__),"..","Data","my_data.csv")
    sheet = pd.read_csv(csv_path)
    date_series = pd.to_datetime(sheet[date_column], format="%m/%d/%Y")
    date_str_list = [dt.strftime("%Y-%m-%d") for dt in date_series]
    return date_str_list


#Function that splits data. The default value is to leave the same number for the
#   correction ratio records
def split_data(data, train_ratio: float = 1-(1155/3708)):
    length = len(data)
    training_number = int(np.floor(train_ratio * length))
    training_data = data[0:training_number]         # up to but not including training_number
    testing_data = data[training_number:]           # from training_number to end
    return training_data, testing_data




