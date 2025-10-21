import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
from skrebate import ReliefF  # Uncomment if you have skrebate installed and want ReliefF
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import mean_squared_error, r2_score  # For evaluation metrics
from mlxtend.feature_selection import SequentialFeatureSelector as SFS  # Import for SFS/SFBS/SFFS/SFBS
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor  # Added for RandomForestRegressor
import zipfile
from sklearn.model_selection import RandomizedSearchCV  # Import for hyperparameter tuning
from sklearn.impute import KNNImputer, SimpleImputer  # New imports for advanced imputation
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS  # <--- This is the new import!


# I chose RESTFULNESS as the predicted measure because it has a high variance of 8.34 and a range
# of 1-10. With 108 responses averaging 6.65 on a 1-10 scale—distributed as
# 0 (0%), 1 (0%), 2 (0%), 3 (5.6%), 4 (9.3%), 5 (11.1%), 6 (15.7%), 7 (21.3%), 8 (23.1%), 9 (12%), and 10 (1.9%)—
# it shows a wide range of sleep experiences. It also has a direct relationship to features such as
# acc_rest_duration and screen_night_activations, which intuitively connect to sleep quality.
# This makes it a suitable and meaningful target for our predictive model.

# Defining a seed to ensure the results are reproducible across multiple runs.
np.random.seed(42)


def load_sensor_data(project_dir):
    """
    Loads raw sensor data from Excel files, concatenates them, and saves
    individual sensor data to CSV files.

    Args:
        project_dir (str): The root directory of the project.

    Returns:
        tuple: A dictionary of DataFrames, one for each sensor type, and the output directory path.
    """
    sensor_dir = os.path.join(project_dir, 'sensor_data')
    output_dir = sensor_dir # Output directory is the same as sensor_dir for intermediate files
    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists

    # List of original Excel files containing sensor data
    files = ['bhq_hisha_2025.xlsx', 'bhq_hisha_2025_s2.xlsx', 'bhq_hisha_2025_s2_additionnal.xlsx',
             'bhq_hisha_2025_s3.xlsx']
    dfs = []
    for file in files: # Looping twice here, should be `for file_name in files:` once
        # Correction: The inner loop `for file in files:` is redundant and should be removed.
        # It causes each file to be processed multiple times.
        file_path = os.path.join(project_dir, 'data', file)
        try:
            df = pd.read_excel(file_path)
            # Ensure 'uid' column exists, which is crucial for identifying participants
            if 'uid' not in df.columns:
                raise KeyError(f"Column 'uid' not found in {file_path}")
            dfs.append(df)
            print(f"Loaded {file} from '{file_path}'")
        except FileNotFoundError:
            print(f"File {file_path} not found. Please ensure all data files are in the 'data' folder.")
            raise # Re-raise the exception to stop execution if a critical file is missing

    # Concatenate all loaded DataFrames into a single sensor_df
    sensor_df = pd.concat(dfs, ignore_index=True)
    # Convert 'datetime' column to datetime objects, coercing errors to NaT (Not a Time)
    sensor_df['datetime'] = pd.to_datetime(sensor_df['datetime'], errors='coerce')
    # Drop rows where 'uid' or 'datetime' are missing, as these are essential for data integrity
    sensor_df = sensor_df.dropna(subset=['uid', 'datetime'])

    # Global filtering: only keep data from a specified start date onwards
    start_date = pd.to_datetime('2025-04-27 10:00:00')
    sensor_df = sensor_df[sensor_df['datetime'] >= start_date]

    # Split the combined DataFrame into separate DataFrames for each sensor type
    sensor_dfs = {}
    sensors = ['accelerometer', 'calls', 'light', 'screen', 'location', 'wifi']
    for sensor in sensors:
        sensor_data = sensor_df[sensor_df['type'] == sensor].copy()
        if sensor_data.empty:
            print(f"No data for {sensor}. Creating empty DataFrame.")
            sensor_dfs[sensor] = pd.DataFrame() # Create an empty DataFrame if no data exists
        else:
            output_path = os.path.join(output_dir, f'{sensor}_data.csv')
            sensor_data.to_csv(output_path, index=False) # Save each sensor's data to a CSV
            print(f"Saved {sensor} data to '{output_path}'")
            sensor_dfs[sensor] = sensor_data # Store the sensor's DataFrame in the dictionary

    return sensor_dfs, output_dir


def load_questionnaire_data(project_dir):
    """
    Loads questionnaire data from CSV files, extracts relevant columns,
    and converts sleep duration strings to hours.

    Args:
        project_dir (str): The root directory of the project.

    Returns:
        pd.DataFrame: A DataFrame containing 'Timestamp', 'uid', and 'sleep_hours'.
    """
    files = ['Session_A_Label.csv', 'Session_B_Label.csv', 'Session_C_Label.csv']
    dfs = []

    for file in files:
        file_path = os.path.join(project_dir, 'DATA', file) # Data files are in 'DATA' subdirectory
        try:
            # Try different encodings as CSV files can vary
            try:
                df = pd.read_csv(file_path, encoding='latin1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='Windows-1252')

            # Validate essential columns
            if 'uid' not in df.columns:
                raise KeyError(f"Column 'uid' not found in {file_path}")
            # Dynamically find the sleep duration column based on its common prefix
            sleep_col = next((col for col in df.columns if col.startswith("How long did you sleep last night")), None)
            if sleep_col is None:
                raise KeyError(f"No sleep duration column found in {file_path}")

            dfs.append(df[['Timestamp', 'uid', sleep_col]]) # Select only necessary columns
        except FileNotFoundError:
            print(f"{file_path} not found. Ensure questionnaire data is in the 'DATA' folder.")
            raise

    quest_df = pd.concat(dfs, ignore_index=True) # Combine all questionnaire data
    # Convert 'Timestamp' to datetime objects, handling errors and assuming day-first format
    quest_df['Timestamp'] = pd.to_datetime(quest_df['Timestamp'], errors='coerce', dayfirst=True)

    def convert_time_to_hours(time_str):
        """
        Helper function to convert 'HH:MM:SS' time strings into total hours (float).
        """
        if pd.isna(time_str):
            return np.nan
        try:
            h, m, s = map(int, str(time_str).split(':')) # Ensure time_str is string before splitting
            return h + m / 60 + s / 3600
        except ValueError: # Catch cases where split or map might fail (e.g., malformed strings)
            return np.nan

    # Apply the conversion to the sleep duration column, which is the last column after selection
    quest_df['sleep_hours'] = quest_df.iloc[:, -1].apply(convert_time_to_hours)
    # Drop the original string column after conversion
    quest_df = quest_df.drop(columns=quest_df.columns[-2]) # Drop the original 'How long did you sleep...' column
    return quest_df


def handle_missing_values(sensor_dfs, output_dir):
    """
    Performs intelligent imputation of missing values for each sensor's DataFrame.
    Strategies are tailored to each sensor type based on typical data characteristics.

    Args:
        sensor_dfs (dict): A dictionary of DataFrames, one for each sensor.
        output_dir (str): Directory to save cleaned data.

    Returns:
        dict: The updated dictionary of DataFrames with missing values handled.
    """
    print("\n--- Starting handle_missing_values ---")
    for sensor in sensor_dfs:
        if sensor_dfs[sensor].empty:
            print(f"Skipping empty {sensor} data.")
            continue

        sensor_data = sensor_dfs[sensor].copy()
        start_date = pd.to_datetime('2025-04-27 10:00:00')
        sensor_data = sensor_data[sensor_data['datetime'] >= start_date]

        initial_nan_count = sensor_data.isna().sum().sum()
        print(f"Sensor: {sensor}, Initial NaNs in raw data: {initial_nan_count}")

        if sensor == 'accelerometer':
            for col in ['x', 'y', 'z']:
                sensor_data[col] = pd.to_numeric(sensor_data[col], errors='coerce') # Convert to numeric, coerce errors
                # Fill NaNs during night hours (0-6 AM) with values representing stillness (9.81 for Z, 0 for X, Y)
                night_mask = sensor_data['datetime'].dt.hour.between(0, 6)
                sensor_data.loc[night_mask & sensor_data[col].isna(), col] = 9.81 if col == 'z' else 0

                # Use forward fill (ffill) and backward fill (bfill) grouped by UID to propagate values
                # This assumes missing values are intermittent and can be inferred from nearby valid readings.
                sensor_data[col] = sensor_data.groupby('uid')[col].ffill()
                sensor_data[col] = sensor_data.groupby('uid')[col].bfill()

                # Fill any remaining NaNs with UID-specific mean, then overall mean, finally 0 as a fallback
                uid_mean_values = sensor_data.groupby('uid')[col].transform('mean')
                sensor_data[col] = sensor_data[col].fillna(uid_mean_values)
                sensor_data[col] = sensor_data[col].fillna(sensor_data[col].mean())
                sensor_data[col] = sensor_data[col].fillna(0)

        elif sensor == 'calls':
            # Convert 'value' to numeric (call duration) and fill NaNs with 0
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce').fillna(0)
            # Impute 'sensor_status': if data contains "MISSED" and status is NaN, assume rejected (2)
            sensor_data['sensor_status'] = sensor_data.apply(
                lambda row: 2 if pd.isna(row['sensor_status']) and 'data' in row and isinstance(row['data'],
                                                                                                str) and '"MISSED"' in
                                 row['data'] else row['sensor_status'], axis=1)
            # Map string statuses to numeric codes
            sensor_data['sensor_status'] = sensor_data['sensor_status'].replace({
                'OUTGOING': 0, 'INCOMING': 1, 'REJECTED': 2
            })
            # Convert status to numeric and fill any remaining NaNs with 2 (rejected/missed as a neutral default)
            sensor_data['sensor_status'] = pd.to_numeric(sensor_data['sensor_status'], errors='coerce').fillna(2)

        elif sensor == 'light':
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce')
            # Cap extremely high light values, assuming they are sensor errors
            sensor_data.loc[sensor_data['value'] > 1e6, 'value'] = 0

            # Use ffill/bfill for light sensor, grouped by UID
            sensor_data['value'] = sensor_data.groupby('uid')['value'].ffill()
            sensor_data['value'] = sensor_data.groupby('uid')['value'].bfill()

            # Fill night NaNs with 0 (assuming it's dark during sleep)
            night_mask = sensor_data['datetime'].dt.hour.between(0, 6)
            sensor_data.loc[night_mask & sensor_data['value'].isna(), 'value'] = 0

            # Fill remaining NaNs with UID-specific mean, then overall mean, finally 10 as a fallback
            uid_mean_values = sensor_data.groupby('uid')['value'].transform('mean')
            sensor_data['value'] = sensor_data['value'].fillna(uid_mean_values)
            sensor_data['value'] = sensor_data['value'].fillna(sensor_data['value'].mean())
            sensor_data['value'] = sensor_data['value'].fillna(10)

        elif sensor == 'screen':
            # Parse 'data' column to infer screen 'on' (1) or 'off' (0) status
            sensor_data['value'] = sensor_data['data'].apply(
                lambda x: 1 if isinstance(x, str) and '"on"' in x else 0 if isinstance(x,
                                                                                       str) and '"off"' in x else np.nan)
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce')

            # Fill night NaNs with 0 (assuming screen is off during sleep)
            night_mask = sensor_data['datetime'].dt.hour.between(0, 6)
            sensor_data.loc[night_mask & sensor_data['value'].isna(), 'value'] = 0

            # Use ffill/bfill for screen value, grouped by UID
            sensor_data['value'] = sensor_data.groupby('uid')['value'].ffill()
            sensor_data['value'] = sensor_data.groupby('uid')['value'].bfill()
            sensor_data['value'] = sensor_data['value'].fillna(0) # Final fallback for any remaining NaNs

        elif sensor == 'location':
            # Fill 'suuid' (location ID) NaNs with 'unknown'
            sensor_data['suuid'] = sensor_data['suuid'].fillna('unknown')
            # Convert 'value' (distance) to numeric and fill NaNs with 0.0 (assuming no movement if missing)
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce').fillna(0.0)
            # Map string statuses to numeric codes
            sensor_data['sensor_status'] = sensor_data['sensor_status'].replace({'ON': 1, 'OFF': 0})
            sensor_data['sensor_status'] = pd.to_numeric(sensor_data['sensor_status'], errors='coerce').fillna(0)
            # Placeholder for 'x' and 'y' coordinates, initialized to 0.0 as they are not used but expected
            sensor_data['x'] = 0.0
            sensor_data['y'] = 0.0

        elif sensor == 'wifi':
            # Fill 'suuid' (WiFi network ID) NaNs with 'NULL'
            sensor_data['suuid'] = sensor_data['suuid'].fillna('NULL')
            # Convert 'level' (signal strength) to numeric and fill with -100 (a common default for no signal)
            sensor_data['level'] = pd.to_numeric(sensor_data.get('level', 0), errors='coerce')
            sensor_data['level'] = sensor_data['level'].fillna(-100.0)

            # Use ffill/bfill for wifi level, grouped by UID
            sensor_data['level'] = sensor_data.groupby('uid')['level'].ffill()
            sensor_data['level'] = sensor_data.groupby('uid')['level'].bfill()

            # Fill remaining NaNs with UID-specific mean, then overall mean, finally -100 as a fallback
            uid_mean_values = sensor_data.groupby('uid')['level'].transform('mean')
            sensor_data['level'] = sensor_data['level'].fillna(uid_mean_values)
            sensor_data['level'] = sensor_data['level'].fillna(sensor_data['level'].mean())
            sensor_data['level'] = sensor_data['level'].fillna(-100)

        output_path = os.path.join(output_dir, f'{sensor}_data_clean.csv')
        sensor_data.to_csv(output_path, index=False)
        print(f"Saved cleaned {sensor} data to '{output_path}'")

        final_nan_count = sensor_data.isna().sum().sum()
        print(f"Sensor: {sensor}, Final NaNs after specific imputation: {final_nan_count}")

        sensor_dfs[sensor] = sensor_data # Update the dictionary with the cleaned DataFrame

    print("--- handle_missing_values completed ---")
    return sensor_dfs


def compute_features(sensor_dfs, quest_df):
    """
    Computes time-series features for each sensor based on daily aggregates
    and specific time windows (e.g., night, pre-sleep). Also computes cross-sensor features.

    Args:
        sensor_dfs (dict): Dictionary of cleaned sensor DataFrames.
        quest_df (pd.DataFrame): Questionnaire data (used conceptually, not directly here for features).

    Returns:
        dict: A dictionary of DataFrames, where each DataFrame contains computed features for a sensor.
    """
    print("\n--- Starting compute_features ---")
    feature_dfs = {}

    for sensor in sensor_dfs:
        if sensor_dfs[sensor].empty:
            print(f"Skipping feature computation for empty {sensor} data.")
            feature_dfs[sensor] = pd.DataFrame()
            continue

        sensor_data = sensor_dfs[sensor].copy()
        start_date = pd.to_datetime('2025-04-27 10:00:00')
        sensor_data = sensor_data[sensor_data['datetime'] >= start_date]
        sensor_data['date'] = sensor_data['datetime'].dt.date # Extract date component
        sensor_data['hour'] = sensor_data['datetime'].dt.hour # Extract hour component
        sensor_data = sensor_data.reset_index(drop=True)

        # Define time masks for different periods of the day
        night_mask = (sensor_data['hour'] >= 22) | (sensor_data['hour'] <= 6) # 10 PM to 6 AM
        pre_sleep_mask = sensor_data['hour'].between(20, 22) # 8 PM to 10 PM
        after_midnight_mask = sensor_data['hour'].between(0, 6) # 12 AM to 6 AM

        features = pd.DataFrame()

        if sensor == 'accelerometer':
            # Calculate magnitude and energy from accelerometer readings
            sensor_data['magnitude'] = np.sqrt(sensor_data['x'] ** 2 + sensor_data['y'] ** 2 + sensor_data['z'] ** 2)
            sensor_data['energy'] = sensor_data['magnitude'] ** 2
            # Identify periods of movement (magnitude significantly different from 9.81 m/s^2, Earth's gravity)
            sensor_data['is_moving'] = sensor_data['magnitude'].apply(lambda x: float(1 if abs(x - 9.81) > 1 else 0))
            # Calculate changes in movement status to detect transitions
            sensor_data['movement_change'] = sensor_data.groupby('uid')['is_moving'].diff().abs().fillna(0).astype(float)
            # Identify significant movements (e.g., sudden jerks)
            sensor_data['significant_movement'] = sensor_data['magnitude'].apply(lambda x: float(1 if x > 10 else 0))

            # Aggregate daily features
            features = sensor_data.groupby(['uid', 'date']).agg(
                acc_magnitude_mean=('magnitude', 'mean'),
                acc_magnitude_std=('magnitude', 'std'),
                acc_magnitude_min=('magnitude', 'min'),
                acc_magnitude_max=('magnitude', 'max'),
                acc_x_mean=('x', 'mean'), acc_x_std=('x', 'std'),
                acc_y_mean=('y', 'mean'), acc_y_std=('y', 'std'),
                acc_z_mean=('z', 'mean'), acc_z_std=('z', 'std'),
                # Sum of energy during night
                acc_night_energy=('energy', lambda x: x[night_mask].sum()),
                # Total movement changes during night
                acc_night_movement_changes=('movement_change', lambda x: float(x[night_mask].sum())),
                # Count of significant movements during night
                acc_night_significant_movements=('significant_movement', lambda x: float(x[night_mask].sum())),
            ).reset_index()

            # Calculate resting periods and wake-up events based on accelerometer data during night
            rest_periods = []
            for uid in sensor_data['uid'].unique():
                uid_data = sensor_data[(sensor_data['uid'] == uid) & night_mask].sort_values('datetime')
                # Determine if the device is static (magnitude close to gravity)
                uid_data['is_static_for_sleep'] = uid_data['magnitude'].apply(lambda x: float(abs(x - 9.81) < 0.5))
                # Detect significant movement events that might indicate waking up
                uid_data['significant_movement_event'] = uid_data['magnitude'].apply(lambda x: 1 if x > 12 else 0)

                for single_date in uid_data['date'].unique():
                    daily_data = uid_data[uid_data['date'] == single_date].copy()

                    longest_static_duration = 0.0
                    if not daily_data[daily_data['is_static_for_sleep'] == 1].empty:
                        # Group consecutive static periods to find the longest one
                        daily_data['static_group'] = (daily_data['is_static_for_sleep'] != daily_data[
                            'is_static_for_sleep'].shift()).cumsum()
                        static_blocks = daily_data[daily_data['is_static_for_sleep'] == 1].groupby('static_group')[
                            'datetime'].agg(['min', 'max'])
                        static_blocks['duration_hours'] = (static_blocks['max'] - static_blocks[
                            'min']).dt.total_seconds() / 3600
                        if not static_blocks.empty:
                            longest_static_duration = static_blocks['duration_hours'].max()

                    wake_up_count = 0.0
                    if not daily_data.empty and daily_data['significant_movement_event'].sum() > 0:
                        # Calculate wake-up events: significant movements separated by at least 5 minutes of no movement
                        wake_up_events_timestamps = []
                        last_movement_time = None
                        for i in range(len(daily_data)):
                            row = daily_data.iloc[i]
                            if row['significant_movement_event'] == 1:
                                if last_movement_time is None or (
                                        row['datetime'] - last_movement_time).total_seconds() / 60 > 5:
                                    wake_up_events_timestamps.append(row['datetime'])
                                last_movement_time = row['datetime']
                        wake_up_count = float(len(wake_up_events_timestamps))

                    # Estimate total rest duration during night based on static samples and typical interval
                    time_interval_seconds = daily_data['datetime'].diff().mode()
                    interval_in_seconds = time_interval_seconds[0].total_seconds() if not time_interval_seconds.empty and \
                                                                                       time_interval_seconds[
                                                                                           0].total_seconds() > 0 else 1.0
                    rest_duration_calc = daily_data[daily_data['is_static_for_sleep'] == 1][
                                             'is_static_for_sleep'].sum() * interval_in_seconds / 3600.0

                    rest_periods.append({
                        'uid': uid,
                        'date': single_date,
                        'rest_duration': rest_duration_calc,
                        'acc_longest_uninterrupted_sleep_duration': longest_static_duration,
                        'acc_night_wake_up_count': float(wake_up_count)
                    })

            rest_df = pd.DataFrame(rest_periods).groupby(['uid', 'date']).agg({
                'rest_duration': 'sum',
                'acc_longest_uninterrupted_sleep_duration': 'max',
                'acc_night_wake_up_count': 'sum'
            }).reset_index()

            # Merge these new rest-related features into the main features DataFrame
            features = features.merge(rest_df, on=['uid', 'date'], how='left')
            # Fill NaNs resulting from the merge (e.g., if no night data for a UID/date) with 0.0
            features['acc_rest_duration'] = features['rest_duration'].fillna(0.0).astype(float)
            features['acc_longest_uninterrupted_sleep_duration'] = features[
                'acc_longest_uninterrupted_sleep_duration'].fillna(0.0).astype(float)
            features['acc_night_wake_up_count'] = features['acc_night_wake_up_count'].fillna(0.0).astype(float)
            features = features.drop(columns=['rest_duration']) # Drop temporary column

        elif sensor == 'calls':
            late_mask = sensor_data['hour'] >= 21 # Define "late" as 9 PM onwards
            # Calculate time remaining until 11 PM (assumed sleep time) for calls before 11 PM
            sensor_data['time_to_sleep'] = sensor_data['datetime'].apply(
                lambda x: (datetime(x.year, x.month, x.day, 23,
                                    0) - x).total_seconds() / 3600 if x.hour < 23 else np.nan)

            # Aggregate daily call features
            features = sensor_data.groupby(['uid', 'date']).agg(
                call_duration_sum=('value', 'sum'), # Total call duration
                call_count=('value', 'count'), # Total number of call events
                incoming_call_count=('sensor_status', lambda x: float((x == 1).sum())) # Count of incoming calls (status 1)
            ).reset_index()

            # Aggregate late-night call features
            late_features = sensor_data[late_mask].groupby(['uid', 'date']).agg(
                late_call_duration=('value', 'sum'),
                late_call_count=('sensor_status', 'count')
            ).reset_index()

            features = features.merge(late_features, on=['uid', 'date'], how='left')
            features['late_call_duration'] = features['late_call_duration'].fillna(0.0).astype(float)
            features['late_call_count'] = features['late_call_count'].fillna(0.0).astype(float)

            # Get the minimum time to sleep (i.e., the latest call before 11 PM)
            last_call = sensor_data.groupby(['uid', 'date'])['time_to_sleep'].min().reset_index(
                name='last_call_to_sleep')
            features = features.merge(last_call, on=['uid', 'date'], how='left')
            features['last_call_to_sleep'] = features['last_call_to_sleep'].fillna(0.0).astype(float)

            # Count missed or rejected calls during night hours
            night_calls_data = sensor_data[night_mask]
            rejected_or_missed_count = night_calls_data.groupby(['uid', 'date'])['sensor_status'].apply(
                lambda x: float((x == 2).sum())
            ).reset_index(name='calls_night_missed_or_rejected_count')

            features = features.merge(rejected_or_missed_count, on=['uid', 'date'], how='left')
            features['calls_night_missed_or_rejected_count'] = features['calls_night_missed_or_rejected_count'].fillna(
                0.0).astype(float)

        elif sensor == 'light':
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce').fillna(0.0).astype(float)
            # Identify periods of high light exposure (e.g., screen light)
            sensor_data['high_light'] = sensor_data['value'].apply(lambda x: float(1.0 if x > 2 else 0.0))
            # Identify periods of low light (e.g., dark room)
            sensor_data['low_light'] = sensor_data['value'].apply(lambda x: float(1.0 if x <= 5 else 0.0))

            # Aggregate daily light features
            features = sensor_data.groupby(['uid', 'date']).agg(
                light_mean=('value', 'mean'),
                light_std=('value', 'std'),
                light_max=('value', 'max'),
                light_min=('value', 'min'),
                # Sum of high light exposure during night
                light_night_high_exposure=('high_light', lambda x: float(x[night_mask].sum())),
                # Proportion of low light during night
                light_night_low_proportion=('low_light', lambda x: float(x[night_mask].mean()))
            ).reset_index()

            # Ensure aggregated columns are float and fill NaNs with 0.0 if no data for a period
            features[['light_mean', 'light_std', 'light_max', 'light_min',
                      'light_night_high_exposure', 'light_night_low_proportion']] = \
                features[['light_mean', 'light_std', 'light_max', 'light_min',
                          'light_night_high_exposure', 'light_night_low_proportion']].fillna(0.0).astype(float)

            # Calculate mean and std of light specifically during night
            night_group = sensor_data[night_mask].groupby(['uid', 'date'])['value']
            features['light_night_mean'] = night_group.mean().reindex(
                features.set_index(['uid', 'date']).index).reset_index(drop=True).fillna(0.0).astype(float)
            features['light_night_std'] = night_group.std().reindex(
                features.set_index(['uid', 'date']).index).reset_index(drop=True).fillna(0.0).astype(float)

            # Count sudden light spikes during night (potential disturbances)
            sudden_spikes_df = sensor_data[night_mask].copy()
            sudden_spikes_df['value_diff'] = sudden_spikes_df.groupby(['uid', 'date'])['value'].diff()
            spikes_count = sudden_spikes_df.groupby(['uid', 'date'])['value_diff'].apply(
                lambda x: float((x > 100).sum()) # Define a spike as a change > 100
            ).reset_index(name='light_night_sudden_spikes_count')

            features = features.merge(spikes_count, on=['uid', 'date'], how='left')
            features['light_night_sudden_spikes_count'] = features['light_night_sudden_spikes_count'].fillna(
                0.0).astype(float)

            # Calculate average light intensity in the pre-sleep period
            pre_sleep_light_avg = sensor_data[pre_sleep_mask].groupby(['uid', 'date'])['value'].mean().reset_index(
                name='light_avg_pre_sleep_intensity')
            features = features.merge(pre_sleep_light_avg, on=['uid', 'date'], how='left')
            features['light_avg_pre_sleep_intensity'] = features['light_avg_pre_sleep_intensity'].fillna(0.0).astype(
                float)

        elif sensor == 'screen':
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce').fillna(0.0).astype(float)
            # Detect screen "on" events (change from 0 to 1)
            sensor_data['screen_change'] = sensor_data.groupby('uid')['value'].diff().eq(1).astype(float)

            # Calculate time remaining until 11 PM if screen is on
            sensor_data['time_to_sleep'] = sensor_data.apply(
                lambda row: (datetime(row['datetime'].year, row['datetime'].month, row['datetime'].day, 23, 0) - row[
                    'datetime']).total_seconds() / 3600
                if row['datetime'].hour < 23 and row['value'] == 1 else np.nan, axis=1)

            # Aggregate daily screen features
            features = sensor_data.groupby(['uid', 'date']).agg(
                screen_on_sum=('value', 'sum'), # Total screen on duration
                screen_event_count=('value', 'count'), # Total screen events
                screen_pre_sleep_sum=('value', lambda x: float(x[pre_sleep_mask].sum())), # Screen on duration pre-sleep
                screen_night_activations=('screen_change', lambda x: float(x[night_mask].sum())) # Screen activation count at night
            ).reset_index()

            # Calculate ratio of night screen time to day screen time
            day_mask = sensor_data['hour'].between(10, 22)
            day_sum = sensor_data[day_mask].groupby(['uid', 'date'])['value'].sum().reindex(
                features.set_index(['uid', 'date']).index, fill_value=0.0
            ).reset_index(drop=True).astype(float)
            night_sum_val = sensor_data[night_mask].groupby(['uid', 'date'])['value'].sum().reindex(
                features.set_index(['uid', 'date']).index, fill_value=0.0
            ).reset_index(drop=True).astype(float)
            features['screen_night_day_ratio'] = (night_sum_val + 1e-10) / (day_sum.clip(lower=1e-5) + 1e-10)
            features.loc[day_sum == 0, 'screen_night_day_ratio'] = 0.0 # Handle division by zero

            # Get the time of the last screen use before 11 PM
            features['screen_last_use_to_sleep'] = sensor_data.groupby(['uid', 'date'])['time_to_sleep'].min().reindex(
                features.set_index(['uid', 'date']).index, fill_value=0.0).reset_index(drop=True).astype(float)

            # Count screen activations after midnight
            after_midnight_activations = sensor_data[after_midnight_mask].groupby(['uid', 'date'])[
                'screen_change'].sum().reset_index(name='screen_activations_after_midnight')
            features = features.merge(after_midnight_activations, on=['uid', 'date'], how='left')
            features['screen_activations_after_midnight'] = features['screen_activations_after_midnight'].fillna(
                0.0).astype(float)

        elif sensor == 'location':
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce').fillna(0.0).astype(float)
            sensor_data['distance'] = sensor_data['value']
            # Identify periods where user is away from home (distance > 50 units)
            sensor_data['away_from_home'] = sensor_data['distance'].apply(lambda x: float(1 if x > 50 else 0))
            # Calculate speed based on distance change over time
            time_diff = sensor_data['datetime'].diff().dt.total_seconds().fillna(1).replace(0, 1)
            sensor_data['speed'] = sensor_data['distance'].diff() / time_diff
            sensor_data['speed'] = sensor_data['speed'].apply(lambda x: max(0, x) if not np.isinf(x) else 0).astype(
                float)
            # Categorize locations into clusters based on distance
            sensor_data['location_cluster'] = sensor_data['distance'].apply(lambda x: float(int(x // 50) + 1))

            # Aggregate daily location features
            features = sensor_data.groupby(['uid', 'date']).agg(
                distance_sum=('distance', 'sum'),
                distance_mean=('distance', 'mean'),
                loc_night_away_from_home=('away_from_home', lambda x: float(x[night_mask].sum())), # Count away from home at night
                loc_pre_sleep_speed=('speed', lambda x: float(x[pre_sleep_mask].mean())), # Average speed pre-sleep
                loc_unique_count=('location_cluster', lambda x: float(x.nunique())) # Number of unique location clusters
            ).reset_index()

            # Fill NaNs from aggregation with 0.0
            features[['distance_sum', 'distance_mean', 'loc_night_away_from_home',
                      'loc_pre_sleep_speed', 'loc_unique_count']] = \
                features[['distance_sum', 'distance_mean', 'loc_night_away_from_home',
                          'loc_pre_sleep_speed', 'loc_unique_count']].fillna(0.0).astype(float)


        elif sensor == 'wifi':
            # Handle 'level' column (WiFi signal strength) - fillna with -100 as default
            sensor_data['level'] = pd.to_numeric(sensor_data.get('level', 0), errors='coerce').fillna(-100.0).astype(
                float)
            sensor_data['suuid'] = sensor_data['suuid'].fillna('NULL').astype(str)

            # Identify the most frequent WiFi network (home network proxy) for each user at night
            home_wifi = sensor_data[night_mask].groupby('uid')['suuid'].agg(
                lambda x: x.mode()[0] if not x.mode().empty else 'NULL').reset_index()
            home_wifi.columns = ['uid', 'home_suuid']
            sensor_data = sensor_data.merge(home_wifi, on='uid', how='left')
            # Check if current WiFi is the identified "home" WiFi
            sensor_data['is_home_wifi'] = (sensor_data['suuid'] == sensor_data['home_suuid']).astype(float)
            # Detect changes in WiFi network (indicating movement or switching networks)
            sensor_data['wifi_change'] = sensor_data.groupby('uid')['suuid'].shift(1).ne(sensor_data['suuid']).astype(
                float)

            # Aggregate daily WiFi features
            features = sensor_data.groupby(['uid', 'date']).agg(
                wifi_level_mean=('level', 'mean'),
                wifi_level_std=('level', 'std'),
                wifi_night_signal_mean=('level', lambda x: float(x[night_mask].mean())), # Average signal strength at night
                wifi_unique_count=('suuid', lambda x: float(x.nunique())), # Number of unique WiFi networks
                wifi_night_changes=('wifi_change', lambda x: float(x[night_mask].sum())), # Count of WiFi network changes at night
                wifi_night_home_time=('is_home_wifi', lambda x: float(x[night_mask].sum())) # Total time connected to home WiFi at night
            ).reset_index()

            # Fill NaNs from aggregation with 0.0
            features[['wifi_level_mean', 'wifi_level_std', 'wifi_night_signal_mean',
                      'wifi_unique_count', 'wifi_night_changes', 'wifi_night_home_time']] = \
                features[['wifi_level_mean', 'wifi_level_std', 'wifi_night_signal_mean',
                          'wifi_unique_count', 'wifi_night_changes', 'wifi_night_home_time']].fillna(0.0).astype(float)

        feature_dfs[sensor] = features # Store the computed features for the current sensor
        sensor_dfs[sensor] = sensor_data # Update sensor_dfs with any modifications (e.g., new columns) to original data

    # --- Cross-sensor features ---
    # These features combine data from multiple sensors to capture more complex behaviors.
    # We need accelerometer, screen, and light data for these.
    acc_df = sensor_dfs.get('accelerometer', pd.DataFrame())
    screen_df = sensor_dfs.get('screen', pd.DataFrame())
    light_df = sensor_dfs.get('light', pd.DataFrame())

    # Check if essential DataFrames are available and have required columns
    if acc_df.empty or 'date' not in acc_df.columns or 'hour' not in acc_df.columns or \
       screen_df.empty or 'date' not in screen_df.columns or 'hour' not in screen_df.columns or \
       light_df.empty or 'date' not in light_df.columns or 'hour' not in light_df.columns:
        print(
            "Warning: One or more required DataFrames (accelerometer, screen, light) are empty or missing 'date' or 'hour' columns. Skipping cross-sensor feature computation.")
        cross_df = pd.DataFrame(columns=['uid', 'date', 'pre_sleep_activity_score', 'night_disturbances'])
        feature_dfs['cross_sensor'] = cross_df
    else:
        # Ensure 'date' columns are datetime objects for merging
        acc_df['date'] = pd.to_datetime(acc_df['date'], errors='coerce')
        screen_df['date'] = pd.to_datetime(screen_df['date'], errors='coerce')
        light_df['date'] = pd.to_datetime(light_df['date'], errors='coerce')

        start_date = pd.to_datetime('2025-04-27 10:00:00')
        acc_df_filtered = acc_df[acc_df['datetime'] >= start_date]
        screen_df_filtered = screen_df[screen_df['datetime'] >= start_date]
        light_df_filtered = light_df[light_df['datetime'] >= start_date]

        cross_features = []
        # Find the common UID-date pairs that exist across all relevant sensor data
        all_uids_dates = pd.merge(acc_df_filtered[['uid', 'date']].drop_duplicates(),
                                  screen_df_filtered[['uid', 'date']].drop_duplicates(),
                                  on=['uid', 'date'], how='inner')
        all_uids_dates = pd.merge(all_uids_dates, light_df_filtered[['uid', 'date']].drop_duplicates(),
                                  on=['uid', 'date'], how='inner')

        for _, row in all_uids_dates.iterrows():
            uid = row['uid']
            date = row['date']

            # Retrieve specific day's data for each sensor
            acc_data = acc_df_filtered[(acc_df_filtered['uid'] == uid) & (acc_df_filtered['date'] == date)].reset_index(
                drop=True)
            screen_data = screen_df_filtered[
                (screen_df_filtered['uid'] == uid) & (screen_df_filtered['date'] == date)].reset_index(drop=True)
            light_data = light_df_filtered[
                (light_df_filtered['uid'] == uid) & (light_df_filtered['date'] == date)].reset_index(drop=True)

            # Re-define time masks for the current day's data to ensure correct indexing
            acc_pre_sleep_mask = acc_data['hour'].between(20, 22) if not acc_data.empty else pd.Series([], dtype=bool)
            acc_night_mask = (acc_data['hour'] >= 22) | (acc_data['hour'] <= 6) if not acc_data.empty else pd.Series([],
                                                                                                                     dtype=bool)
            screen_pre_sleep_mask = screen_data['hour'].between(20, 22) if not screen_data.empty else pd.Series([],
                                                                                                                dtype=bool)
            screen_night_mask = (screen_data['hour'] >= 22) | (
                    screen_data['hour'] <= 6) if not screen_data.empty else pd.Series([], dtype=bool)
            light_pre_sleep_mask = light_data['hour'].between(20, 22) if not light_data.empty else pd.Series([],
                                                                                                             dtype=bool)
            light_night_mask = (light_data['hour'] >= 22) | (
                    light_data['hour'] <= 6) if not light_data.empty else pd.Series([], dtype=bool)

            # Calculate 'pre_sleep_activity_score'
            # This score combines accelerometer energy, screen time, and average light level during pre-sleep
            acc_energy = float(
                acc_data.loc[acc_pre_sleep_mask, 'magnitude'].apply(lambda x: x ** 2).sum() if not acc_data.loc[
                    acc_pre_sleep_mask].empty else 0.0)
            screen_time = float(screen_data.loc[screen_pre_sleep_mask, 'value'].sum() if not screen_data.loc[
                screen_pre_sleep_mask].empty else 0.0)
            light_level = float(light_data.loc[light_pre_sleep_mask, 'value'].mean() if not light_data.loc[
                light_pre_sleep_mask].empty else 0.0)
            activity_score = float((acc_energy / (acc_energy + 1e-10)) + screen_time + (
                    light_level / 100)) # Small constant added to avoid division by zero

            # Calculate 'night_disturbances'
            # This score combines accelerometer movement, screen activations, and light disturbances during night
            acc_disturb = float(
                acc_data.loc[acc_night_mask, 'magnitude'].apply(lambda x: 1 if abs(x - 9.81) > 1 else 0).sum() if not
                acc_data.loc[acc_night_mask].empty else 0.0)
            screen_disturb = float(
                screen_data.loc[screen_night_mask, 'value'].diff().eq(1).sum() if not screen_data.loc[
                    screen_night_mask].empty else 0.0)
            light_disturb = float(
                light_data.loc[light_night_mask, 'value'].apply(lambda x: 1 if x > 10 else 0).sum() if not
                light_data.loc[light_night_mask].empty else 0.0)
            disturbances = float(acc_disturb + screen_disturb + light_disturb)

            cross_features.append({
                'uid': uid,
                'date': date,
                'pre_sleep_activity_score': activity_score,
                'night_disturbances': disturbances
            })

        cross_df = pd.DataFrame(cross_features)
        # Fill any NaNs that might have arisen from empty daily data for a specific UID/date with 0.0
        cross_df[['pre_sleep_activity_score', 'night_disturbances']] = cross_df[
            ['pre_sleep_activity_score', 'night_disturbances']].fillna(0.0)

        feature_dfs['cross_sensor'] = cross_df

    print("Summary of computed features for all sensors:")
    for sensor in feature_dfs:
        if not feature_dfs[sensor].empty:
            print(f"Features for {sensor}: {feature_dfs[sensor].columns.tolist()}")
            nan_count_computed = feature_dfs[sensor].isna().sum().sum()
            print(f"  NaNs in computed features for {sensor}: {nan_count_computed}")
        else:
            print(f"No features computed for {sensor} (empty DataFrame).")

    print("--- compute_features completed ---")
    return feature_dfs


def merge_features(feature_dfs, quest_df, output_dir):
    """
    Merges all computed features from different sensors into a single DataFrame.
    Applies advanced imputation techniques (KNNImputer followed by SimpleImputer)
    to handle any remaining missing values in the merged feature set.

    Args:
        feature_dfs (dict): A dictionary of DataFrames, each containing features for a specific sensor.
        quest_df (pd.DataFrame): Questionnaire data (primarily for 'sleep_hours' as a feature).
        output_dir (str): Directory to save the merged and cleaned feature DataFrame.

    Returns:
        dict: A dictionary containing the single merged feature DataFrame under the key 'all'.
    """
    print("\n--- Starting merge_features ---")
    all_features = pd.DataFrame()
    for sensor in feature_dfs:
        if not feature_dfs[sensor].empty:
            feature_dfs[sensor]['date'] = pd.to_datetime(feature_dfs[sensor]['date'], errors='coerce')
            if all_features.empty:
                all_features = feature_dfs[sensor].copy()
            else:
                # Merge DataFrames on 'uid' and 'date' to combine features
                temp_df = feature_dfs[sensor].set_index(['uid', 'date'])
                all_features = pd.concat([all_features.set_index(['uid', 'date']), temp_df], axis=1,
                                         join='outer').reset_index()
                all_features = all_features.loc[:, ~all_features.columns.duplicated()] # Drop duplicate columns

    # If, for some reason, 'uid' or 'date' columns were lost, try to recover (shouldn't happen with reset_index)
    if 'uid' not in all_features.columns or 'date' not in all_features.columns:
        all_features = all_features.rename(columns={'index': 'uid'}).reset_index(drop=True)
        if 'date' not in all_features.columns:
            all_features['date'] = pd.NaT

    # Initial filling for certain calls-related features that semantically should be 0 if missing
    calls_columns_to_fill_zero = ['call_duration_sum', 'call_count', 'incoming_call_count', 'late_call_duration',
                                  'late_call_count', 'last_call_to_sleep', 'calls_night_missed_or_rejected_count']
    for col in calls_columns_to_fill_zero:
        if col in all_features.columns:
            all_features[col] = all_features[col].fillna(0.0)

    print(f"Total NaNs in all_features BEFORE advanced imputation: {all_features.isna().sum().sum()}")
    print(f"NaN count per column BEFORE advanced imputation (top 10):")
    print(all_features.isna().sum().sort_values(ascending=False).head(10))

    # --- Apply advanced imputation (KNNImputer then SimpleImputer) ---
    # Select only numerical columns for imputation
    numerical_cols = all_features.select_dtypes(include=[np.number]).columns.tolist()

    # Exclude 'uid' from imputation if it's a numeric ID and not a feature
    if 'uid' in numerical_cols:
        numerical_cols.remove('uid')

    original_index = all_features.index
    original_columns = all_features.columns

    # 1. KNNImputer: Fills missing values using the k-nearest neighbors approach.
    # This is often more sophisticated than simple mean imputation as it considers feature relationships.
    if not numerical_cols:
        print("No numerical columns found for advanced imputation.")
    else:
        print(f"Applying KNNImputer on {len(numerical_cols)} numerical columns...")
        # Adjust n_neighbors to avoid errors if dataset is too small
        n_neighbors_knn = min(5, all_features.shape[0] - 1)
        if n_neighbors_knn < 1:
            print("Not enough samples for KNNImputer. Skipping KNNImputer.")
            knn_imputed_data = all_features[numerical_cols].values
        else:
            knn_imputer = KNNImputer(n_neighbors=n_neighbors_knn)
            knn_imputed_data = knn_imputer.fit_transform(all_features[numerical_cols])

        # Convert the imputed numpy array back to a DataFrame, preserving column names
        all_features_imputed_knn = pd.DataFrame(knn_imputed_data, columns=numerical_cols, index=original_index)

        # Overwrite original numerical columns with the KNN-imputed ones
        for col in numerical_cols:
            all_features[col] = all_features_imputed_knn[col]

    # 2. SimpleImputer: Used as a fallback for any remaining NaNs in numerical columns
    # (e.g., if a column was entirely NaN and KNNImputer couldn't fill it).
    print("Applying SimpleImputer (mean strategy) as fallback for any remaining numerical NaNs...")
    simple_imputer = SimpleImputer(strategy='mean')

    numerical_cols_with_nan_after_knn = all_features[numerical_cols].columns[
        all_features[numerical_cols].isna().any()].tolist()
    if numerical_cols_with_nan_after_knn:
        all_features[numerical_cols_with_nan_after_knn] = simple_imputer.fit_transform(
            all_features[numerical_cols_with_nan_after_knn])
        print(f"SimpleImputer filled NaNs in: {numerical_cols_with_nan_after_knn}")
    else:
        print("No numerical NaNs left after KNNImputer, skipping SimpleImputer for numerical columns.")

    print(f"Total NaNs in all_features AFTER advanced imputation: {all_features.isna().sum().sum()}")
    print(f"NaN count per column AFTER advanced imputation (top 10):")
    print(all_features.isna().sum().sort_values(ascending=False).head(10))

    # --- Drop rows with any remaining NaNs (primarily non-numerical columns or if imputation failed for some edge cases) ---
    initial_rows = all_features.shape[0]
    all_features_cleaned = all_features.dropna(
        how='any').copy()
    rows_dropped = initial_rows - all_features_cleaned.shape[0]
    print(f"Dropped {rows_dropped} rows with any remaining NaN values after all imputation steps.")

    output_path = os.path.join(output_dir, 'all_features.csv')
    all_features_cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned all features to '{output_path}'")

    print("--- merge_features completed ---")
    return {'all': all_features_cleaned}


# This function appears twice. Keeping the second one as it aligns with the main execution block logic.
def load_data(project_dir):
    """
    Loads the pre-computed all_features.csv and merges it with questionnaire data
    to include the 'restfulness' target variable and 'sleep_hours' as a potential baseline feature.

    Args:
        project_dir (str): The root directory of the project.

    Returns:
        tuple: A tuple containing the merged DataFrame and the output directory path.
    """
    output_dir = os.path.join(project_dir, 'sensor_data')
    features_path = os.path.join(output_dir, 'all_features.csv')
    all_features = pd.read_csv(features_path)
    # print(all_features) # Commented out verbose print, can uncomment for debugging
    all_features['date'] = pd.to_datetime(all_features['date'], format='%Y-%m-%d', errors='coerce')
    all_features['uid'] = pd.to_numeric(all_features['uid'], errors='coerce')
    print(
        f"all_features shape: {all_features.shape}, columns: {all_features.columns.tolist()}, sample dates: {all_features['date'].head().tolist()}")
    # print(all_features) # Commented out verbose print

    # Load questionnaire data for 'restfulness' and 'sleep_hours'
    quest_files = ['Session_A_Label.csv', 'Session_B_Label.csv', 'Session_C_Label.csv']
    quest_dfs = []
    for file in quest_files:
        file_path = os.path.join(project_dir, 'DATA', file)
        try:
            try:
                df = pd.read_csv(file_path, encoding='latin1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='Windows-1252')

            if 'uid' not in df.columns or 'Timestamp' not in df.columns:
                raise KeyError(f"Missing 'uid' or 'Timestamp' in {file}")

            # Identify the sleep duration column dynamically
            sleep_duration_col_name = next(
                (col for col in df.columns if col.startswith("How long did you sleep last night")), None)
            if sleep_duration_col_name is None:
                raise KeyError(f"No sleep duration column found in {file_path}")

            # Helper function to convert time string to hours
            def convert_time_to_hours(time_str):
                if pd.isna(time_str):
                    return np.nan
                try:
                    h, m, s = map(int, str(time_str).split(':'))
                    return h + m / 60 + s / 3600
                except ValueError:
                    return np.nan

            df['sleep_hours'] = df[sleep_duration_col_name].apply(convert_time_to_hours)

            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%m/%d/%Y', errors='coerce')
            df['date'] = df['Timestamp'] # Rename 'Timestamp' to 'date' for consistent merging
            df['uid'] = pd.to_numeric(df['uid'], errors='coerce')

            # Select relevant columns, including 'restfulness' and 'sleep_hours'
            restfulness_col = "How restful was your sleep?"
            if restfulness_col not in df.columns:
                raise KeyError(
                    f"No restfulness column '{restfulness_col}' found in {file_path}. Available columns: {df.columns.tolist()}")

            quest_dfs.append(df[['uid', 'date', restfulness_col, 'sleep_hours']])
            print(f"Loaded {file} shape: {df.shape}, sample dates: {df['date'].head().tolist()}")
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            raise

    quest_df = pd.concat(quest_dfs, ignore_index=True)
    quest_df = quest_df.rename(columns={restfulness_col: 'restfulness'}) # Standardize column name

    # Merge all_features with questionnaire data on 'uid' and 'date'
    merged_data = pd.merge(all_features, quest_df, on=['uid', 'date'], how='left')

    # Ensure 'restfulness' and 'sleep_hours' are numeric after merge
    merged_data['restfulness'] = pd.to_numeric(merged_data['restfulness'], errors='coerce')
    merged_data['sleep_hours'] = pd.to_numeric(merged_data['sleep_hours'], errors='coerce')

    # Remove rows where 'restfulness' is NaN, as it is our primary target variable
    merged_data = merged_data.dropna(subset=['restfulness'])
    print(
        f"merged_data after dropna shape: {merged_data.shape}, sample restfulness: {merged_data['restfulness'].head().tolist()}")

    return merged_data, output_dir


def prepare_and_split_data(merged_data, output_dir):
    """
    Prepares the data for modeling by separating features (X) and target (Y),
    and then splitting it into training and testing sets *per UID* while
    ensuring consistent scaling.

    Args:
        merged_data (pd.DataFrame): The full merged dataset with features and labels.
        output_dir (str): Directory to save the split datasets.

    Returns:
        tuple: (X_train_all_scaled, X_test_all_scaled, y_train_all, y_test_all)
               Scaled training features, scaled testing features, training labels, testing labels.
               Returns None if data is insufficient for splitting.
    """
    # Set a random seed for reproducibility. This is crucial for ensuring the same train/test split every time.
    np.random.seed(42)

    uids = merged_data['uid'].unique()
    X_features_all = pd.DataFrame() # To collect all features (before per-UID split)
    Y_labels_all = pd.Series(dtype='float64') # To collect all labels (before per-UID split)

    print(f"DEBUG: prepare_and_split_data - Initial merged_data shape: {merged_data.shape}")
    print(f"DEBUG: prepare_and_split_data - Unique UIDs: {len(uids)}")

    # Initialize empty DataFrames and Series to store combined train/test data from all UIDs
    X_train_all = pd.DataFrame()
    X_test_all = pd.DataFrame()
    y_train_all = pd.Series(dtype='float64')
    y_test_all = pd.Series(dtype='float64')

    scaler = StandardScaler() # Initialize the scaler for feature normalization

    all_uids_contributed_to_split = False # Flag to track successful UID contributions

    # Iterate through each unique UID to create per-UID train/test splits
    for uid in uids:
        uid_data = merged_data[merged_data['uid'] == uid].sort_values('date')

        # Drop identifier columns and the target variable to get features
        features = uid_data.drop(columns=['uid', 'date', 'restfulness'])
        target = uid_data['restfulness']

        # Select only numerical columns for feature processing
        features = features.select_dtypes(include=[np.number])
        # Drop rows with any NaNs *within the features* for the current UID before splitting
        valid_indices = features.notna().all(axis=1)
        features = features[valid_indices].reset_index(drop=True)
        target = target[valid_indices].reset_index(drop=True)

        # Skip UID if not enough data remains after NaN filtering
        if len(features) == 0:
            print(
                f"DEBUG: prepare_and_split_data - UID {uid} has no features after NaN clean for splitting. Skipping UID.")
            continue
        if len(features) < 2:
            print(
                f"DEBUG: prepare_and_split_data - UID {uid} has less than 2 samples ({len(features)}). Cannot split train/test for this UID. Skipping.")
            continue

        # Calculate train/test split size (80/20)
        train_size = int(0.8 * len(features))
        if train_size == 0 or len(features) - train_size == 0: # Ensure both train and test sets are non-empty
            print(
                f"DEBUG: prepare_and_split_data - UID {uid} has too few samples ({len(features)}) for an 80/20 split. Skipping.")
            continue

        # Randomly select indices for train and test sets
        train_indices = np.random.choice(len(features), train_size, replace=False)
        test_indices = np.setdiff1d(np.arange(len(features)), train_indices)

        X_train_uid = features.iloc[train_indices]
        X_test_uid = features.iloc[test_indices]
        y_train_uid = target.iloc[train_indices].reset_index(drop=True)
        y_test_uid = target.iloc[test_indices].reset_index(drop=True)
        # Apply scaling inside this function
        # StandardScaler is ideal for my dataset because it standardizes features with different units and scales, making them comparable.
        # It is more robust to outliers than MinMax scaling,
        # which can be skewed by extreme values. This helps improve regression model performance and stability.
        X_train_scaled_uid = scaler.fit_transform(X_train_uid)
        X_test_scaled_uid = scaler.transform(X_test_uid)

        # Convert scaled numpy arrays back to DataFrames, preserving column names and original index
        X_train_scaled_uid = pd.DataFrame(X_train_scaled_uid, columns=X_train_uid.columns, index=X_train_uid.index)
        X_test_scaled_uid = pd.DataFrame(X_test_scaled_uid, columns=X_test_uid.columns, index=X_test_uid.index)

        # Concatenate per-UID scaled data into overall train/test sets
        X_train_all = pd.concat([X_train_all, X_train_scaled_uid], ignore_index=True)
        X_test_all = pd.concat([X_test_all, X_test_scaled_uid], ignore_index=True)
        y_train_all = pd.concat([y_train_all, y_train_uid], ignore_index=True).squeeze() # .squeeze() ensures it remains a Series
        y_test_all = pd.concat([y_test_all, y_test_uid], ignore_index=True).squeeze()

        all_uids_contributed_to_split = True # Mark that at least one UID successfully contributed

    print(f"DEBUG: prepare_and_split_data - Final X_train_all shape: {X_train_all.shape}")
    print(f"DEBUG: prepare_and_split_data - Final X_test_all shape: {X_test_all.shape}")

    # Check if any data was collected for train/test sets, otherwise return None
    if X_train_all.empty or X_test_all.empty or not all_uids_contributed_to_split:
        print("No sufficient data to create train/test sets for all UIDs.")
        print("DEBUG: prepare_and_split_data - Returning None.")
        return None

    # Save the aggregated (but still distinct) train/test sets to CSV files
    X_features_all.to_csv(os.path.join(output_dir, 'X_features.csv'), index=False)
    Y_labels_all.to_csv(os.path.join(output_dir, 'Y_labels.csv'), index=False, header=True)
    X_train_all.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_all.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train_all.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False, header=True)
    y_test_all.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False, header=True)
    print(f"Saved global matrices to {output_dir}")

    return X_train_all, X_test_all, y_train_all, y_test_all


# VETTING FUNCTION (Applied after train/test split, on already scaled data)
def vetting(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
            target_column_name: str = 'restfulness', num_features_to_keep: int = 15,
            outlier_threshold_iqr: float = 1.5, mi_threshold_for_initial_selection: float = 0.003,
            mi_threshold_for_redundancy: float = 0.5):
    """
    Performs vetting on the training and testing datasets separately (but consistently)
    after they have been split and normalized.
    Steps include:
    1. Outlier Treatment (Capping) on X_train. The same capping values are applied to X_test.
    2. Select a larger pool of top features based on Mutual Information with the target (learned from X_train).
    3. Remove redundant features from this pool using a combination of Feature-Feature MI and Spearman correlation with the target.
    4. Final selection of exactly num_features_to_keep features using ReliefF.
    5. Apply the selected features to both X_train and X_test.

    Args:
        X_train (pd.DataFrame): The training features DataFrame (expected to be already scaled).
        X_test (pd.DataFrame): The testing features DataFrame (expected to be already scaled).
        y_train (pd.Series): The training target Series.
        y_test (pd.Series): The testing target Series.
        target_column_name (str): The name of the target column (for conceptual clarity, not directly used for column selection here).
        num_features_to_keep (int): The exact number of features to keep at the end of the process.
        outlier_threshold_iqr (float): Multiplier for the IQR to define outlier bounds (e.g., 1.5 for standard box plot).
        mi_threshold_for_initial_selection (float): Threshold for initial selection of features relevant to the target.
                                                     Features with MI < this threshold are excluded early.
        mi_threshold_for_redundancy (float): Threshold for detecting redundancy using MI between features.

    Returns:
        tuple: (X_train_vetted, X_test_vetted, y_train, y_test, selected_feature_names)
               DataFrames with vetted features and the original target Series.
               If vetting fails, returns original X_train/X_test and an empty list of features.
    """
    print(
        f"\n--- Starting Vetting Process (after train/test split): Aiming for EXACTLY {num_features_to_keep} features ---")
    print(f"Initial features in X_train: {X_train.columns.tolist()}")

    # IMPORTANT: Remove 'sleep_hours' from features if present, as it is a baseline and not a sensor-derived feature.
    # This prevents it from being part of the feature selection for the main model.
    if 'sleep_hours' in X_train.columns:
        print("Removing 'sleep_hours' from features for Vetting process as per instruction.")
        X_train = X_train.drop(columns=['sleep_hours'])
        X_test = X_test.drop(columns=['sleep_hours'])

    # Ensure y_train is a Series, which is required for Mutual Information calculation
    if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    # --- Step 1: Outlier Treatment (Capping) ---
    print("--- Step 1: Outlier Treatment (Capping) ---")
    X_train_outlier_treated = X_train.copy()
    X_test_outlier_treated = X_test.copy()

    for col in X_train.columns:
        # Calculate IQR (Interquartile Range) from the training data to define outlier bounds
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - outlier_threshold_iqr * IQR
        upper_bound = Q3 + outlier_threshold_iqr * IQR

        # Apply capping: values below lower_bound are set to lower_bound, values above upper_bound are set to upper_bound.
        # This is done on both training and testing data using bounds derived *only* from training data to avoid data leakage.
        X_train_outlier_treated[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test_outlier_treated[col] = np.clip(X_test[col], lower_bound, upper_bound)
    print("Outlier capping applied based on IQR from training data.")

    # Identify and remove constant features after outlier treatment.
    # Constant features (those with only one unique value) provide no information and can cause issues in feature selection.
    non_constant_features_train = X_train_outlier_treated.columns[X_train_outlier_treated.nunique() > 1]
    if non_constant_features_train.empty:
        print(
            "No non-constant features in training data after outlier treatment. Cannot perform vetting. Returning original data.")
        return X_train, X_test, y_train, y_test, []

    X_train_filtered_for_mi = X_train_outlier_treated[non_constant_features_train]
    X_test_filtered_for_mi = X_test_outlier_treated[non_constant_features_train] # Apply the same filter to test set

    # --- Step 2: Feature-Label Relevance (Mutual Information) ---
    # Mutual Information (MI) measures the dependency between two variables.
    # Here, we calculate MI between each feature and the target variable (restfulness).
    print("\n--- Step 2: Feature-Label Relevance (Mutual Information) ---")
    print(
        f"Calculating Feature-Label Mutual Information on training data for {len(X_train_filtered_for_mi.columns)} features...")
    # `mutual_info_regression` expects numpy arrays.
    mi_scores_target = mutual_info_regression(X_train_filtered_for_mi.values, y_train)
    mi_scores_target_series = pd.Series(mi_scores_target, index=X_train_filtered_for_mi.columns)
    mi_scores_target_series = mi_scores_target_series.sort_values(ascending=False)

    # Select an initial pool of features that have MI above a certain threshold, and take a maximum number.
    # This acts as a first pass to filter out irrelevant features.
    initial_pool_size = max(num_features_to_keep * 2, min(X_train_filtered_for_mi.shape[1], 30))
    pre_selected_features = mi_scores_target_series[mi_scores_target_series >= mi_threshold_for_initial_selection].head(
        initial_pool_size).index.tolist()

    if not pre_selected_features:
        print(
            f"No features passed initial MI threshold ({mi_threshold_for_initial_selection}) or initial pool empty. Cannot proceed with vetting. Returning original data.")
        return X_train, X_test, y_train, y_test, []

    print(
        f"Initially selected a pool of {len(pre_selected_features)} features based on Feature-Label MI >= {mi_threshold_for_initial_selection}.")
    print("Feature-Label MI Scores (Top 10 of initial pool):")
    print(mi_scores_target_series[pre_selected_features].head(10))

    # --- Step 3: Redundancy Removal (Feature-Feature MI + Spearman for decision) ---
    # This step aims to remove features that are highly correlated with other features,
    # as redundant features can increase model complexity without adding much predictive power.
    print("\n--- Step 3: Redundancy Removal (Feature-Feature MI + Spearman for decision) ---")
    features_after_redundancy_removal = list(pre_selected_features)

    if len(features_after_redundancy_removal) < 2:
        print("Not enough features for redundancy check (less than 2). Skipping redundancy removal.")
    else:
        features_for_redundancy_df = X_train_outlier_treated[features_after_redundancy_removal]
        # Filter out constant features within this subset for MI calculation
        features_for_redundancy_df_filtered = features_for_redundancy_df.loc[:,
                                              features_for_redundancy_df.nunique() > 1]

        if features_for_redundancy_df_filtered.empty:
            print("All features in the redundancy check subset are constant. Skipping redundancy removal.")
        else:
            # Calculate Feature-Feature MI matrix to detect redundancy between features
            mi_matrix_ff = pd.DataFrame(index=features_for_redundancy_df_filtered.columns,
                                        columns=features_for_redundancy_df_filtered.columns)

            for i in mi_matrix_ff.columns:
                for j in mi_matrix_ff.columns:
                    if i == j:
                        mi_matrix_ff.loc[i, j] = 0.0 # MI of a feature with itself is not useful here
                    else:
                        mi_matrix_ff.loc[i, j] = mutual_info_regression(
                            features_for_redundancy_df_filtered[[i]].values,
                            features_for_redundancy_df_filtered[j].values
                        )[0]
            mi_matrix_ff = mi_matrix_ff.astype(float)
            print("Feature-Feature MI Matrix computed (sample top-left corner):")
            print(mi_matrix_ff.head(5).iloc[:, :5])

            # Calculate Spearman correlation with the target.
            # This helps decide which feature to keep when two features are redundant:
            # we keep the one that is more strongly correlated with the target.
            spearman_corr_target = {}
            for col in features_for_redundancy_df_filtered.columns:
                if X_train_outlier_treated[col].nunique() > 1 and y_train.nunique() > 1:
                    corr, _ = spearmanr(X_train_outlier_treated[col], y_train)
                    spearman_corr_target[col] = abs(corr) # Use absolute correlation for strength
                else:
                    spearman_corr_target[col] = 0.0 # Assign 0 if constant

            spearman_series = pd.Series(spearman_corr_target)

            # Sort features by their relevance to the target (MI first, then Spearman as tie-breaker)
            relevance_ranking = mi_scores_target_series[features_after_redundancy_removal].sort_values(ascending=False)

            # Iteratively remove redundant features
            to_remove_in_this_step = set()
            current_candidates = list(relevance_ranking.index)

            idx = 0
            while idx < len(current_candidates):
                f1_name = current_candidates[idx]
                jdx = idx + 1
                while jdx < len(current_candidates):
                    f2_name = current_candidates[jdx]

                    # Check if MI between f1 and f2 is above the redundancy threshold
                    if f1_name in mi_matrix_ff.columns and f2_name in mi_matrix_ff.columns and mi_matrix_ff.loc[
                        f1_name, f2_name] >= mi_threshold_for_redundancy:
                        # Decide which feature to remove: the one with lower Spearman correlation to the target
                        corr_f1 = spearman_series.get(f1_name, 0.0)
                        corr_f2 = spearman_series.get(f2_name, 0.0)

                        if corr_f1 < corr_f2:
                            if f1_name not in to_remove_in_this_step:
                                to_remove_in_this_step.add(f1_name)
                                current_candidates.pop(idx) # Remove from list and re-evaluate current index
                                idx -= 1
                            break # Break inner loop, restart outer for current idx (new feature)
                        else:
                            if f2_name not in to_remove_in_this_step:
                                to_remove_in_this_step.add(f2_name)
                                current_candidates.pop(jdx)
                            jdx -= 1
                    jdx += 1
                idx += 1

            features_after_redundancy_removal = [f for f in pre_selected_features if f not in to_remove_in_this_step]
            # Re-order the remaining features based on their initial relevance ranking
            features_after_redundancy_removal = relevance_ranking.index.intersection(
                features_after_redundancy_removal).tolist()

    print(f"Features after redundancy removal: {len(features_after_redundancy_removal)} features.")
    if len(features_after_redundancy_removal) > 0:
        print("Top 10 features after redundancy removal:")
        print(features_after_redundancy_removal[:10])
    else:
        print("No features remained after redundancy removal.")

    # --- Step 4: Final selection using ReliefF ---
    # ReliefF is a feature weighting algorithm that assigns scores to features based on their ability to
    # distinguish between nearby instances of different classes/values. It's robust to feature interactions.
    print("\n--- Step 4: Final selection using ReliefF ---")
    if not features_after_redundancy_removal:
        print("No features remain after redundancy removal. Cannot perform ReliefF. Returning original data.")
        return X_train, X_test, y_train, y_test, []

    # Prepare data for ReliefF: ensure only selected and non-constant features are included
    X_train_for_relief = X_train_outlier_treated[features_after_redundancy_removal]
    X_train_for_relief = X_train_for_relief.loc[:, X_train_for_relief.nunique() > 1]

    if X_train_for_relief.empty:
        print("No non-constant features remain for ReliefF. Returning original data.")
        return X_train, X_test, y_train, y_test, []

    # Adjust `n_features_to_select` for ReliefF if fewer features are available than desired
    n_features_for_relief = min(num_features_to_keep, X_train_for_relief.shape[1])
    if n_features_for_relief == 0:
        print("No features available for ReliefF to select. Returning original data.")
        return X_train, X_test, y_train, y_test, []

    # Ensure the target variable has variance, which is crucial for ReliefF
    if y_train.nunique() < 2:
        print("Target 'y_train' has no variance. ReliefF cannot be applied. Returning original data.")
        return X_train, X_test, y_train, y_test, []

    # Initialize ReliefF. `n_neighbors` should be less than the number of samples.
    relief = ReliefF(n_features_to_select=n_features_for_relief,
                     n_neighbors=min(20, len(X_train_for_relief) - 1))

    if len(X_train_for_relief) <= relief.n_neighbors:
        print(
            f"Not enough samples ({len(X_train_for_relief)}) for ReliefF with n_neighbors={relief.n_neighbors}. Adjusting n_neighbors to {len(X_train_for_relief) - 1}.")
        relief.n_neighbors = max(1, len(X_train_for_relief) - 1)

    if relief.n_neighbors < 1:
        print("Not enough samples for ReliefF (n_neighbors cannot be less than 1). Returning original data.")
        return X_train, X_test, y_train, y_test, []

    relief.fit(X_train_for_relief.values, y_train.values) # Fit ReliefF to get feature importances

    # Get the names of the top features selected by ReliefF
    selected_indices_relief = relief.top_features_[:n_features_for_relief]
    selected_feature_names = X_train_for_relief.columns[selected_indices_relief].tolist()

    if len(selected_feature_names) != num_features_to_keep:
        print(
            f"WARNING: Final number of features selected by ReliefF is {len(selected_feature_names)}, not exactly {num_features_to_keep}. This is due to limited relevant or non-redundant features.")

    # Apply the finally selected features to both training and testing sets
    X_train_vetted = X_train_outlier_treated[selected_feature_names].copy()
    X_test_vetted = X_test_outlier_treated[selected_feature_names].copy()

    if X_train_vetted.empty or X_test_vetted.empty:
        print("Vetted X_train or X_test became empty after final feature selection. Returning original data.")
        return X_train, X_test, y_train, y_test, []

    print(f"Final X_train shape after vetting: {X_train_vetted.shape}")
    print(f"Final X_test shape after vetting: {X_test_vetted.shape}")
    print(f"Selected features: {selected_feature_names}")
    print(f"--- Vetting Process Completed ---")

    return X_train_vetted, X_test_vetted, y_train, y_test, selected_feature_names


def feature_selection_wrapper(X_train_vetted: pd.DataFrame, X_test_vetted: pd.DataFrame,
                              y_train: pd.Series, y_test: pd.Series,
                              output_dir: str,
                              # Using RandomForestRegressor with default parameters for speed and good general performance
                              model_for_efs=RandomForestRegressor(random_state=42, n_estimators=50, max_depth=8),
                              scoring_metric: str = 'neg_mean_squared_error',
# We use Exhaustive Feature Selection (EFS) with RandomForestRegressor to find the optimal
# feature subset. EFS evaluates every possible combination within the specified range.
# The choice of selecting exactly 10 features (k_features_to_select=(10, 10)) is a crucial
# trade-off for computational feasibility. While EFS guarantees finding the best subset for
# a given size, its exponential complexity means evaluating a wider range or a slightly
# a given size, its exponential complexity means evaluating a wider range or a slightly
# a larger fixed number of features (e.g., 13) would lead to prohibitively long execution times.
# Therefore, fixing at 10 features allows us to leverage EFS's thoroughness without
# making the process impractical.
                              k_features_to_select=(10, 10),
                              cv: int = 5): # Cross-validation folds for evaluating feature subsets
    """
    Performs Feature Selection using a Wrapper Method (Exhaustive Feature Selector - EFS).
    EFS exhaustively evaluates all possible feature combinations within a specified range
    to find the optimal subset that maximizes the chosen scoring metric.

    Args:
    X_train_vetted (pd.DataFrame): Vetted and scaled training features.
    X_test_vetted (pd.DataFrame): Vetted and scaled testing features.
    y_train (pd.Series): Training labels.
    y_test (pd.Series): Testing labels.
    output_dir (str): Directory to save output files.
    model_for_efs: The machine learning model to use inside the EFS for evaluation.
    scoring_metric (str): The metric to optimize (e.g., 'neg_mean_squared_error' for MSE, 'r2' for R-squared).
    k_features_to_select (tuple): (min_features, max_features) to select. For example, (10, 10) means exactly 10 features.
    cv (int): Number of cross-validation folds for EFS.

    Returns:
    tuple: (X_train_final, X_test_final, y_train_final, y_test_final, selected_feature_names)
    DataFrames with the final selected features and corresponding labels.
    """
    method_name = "Exhaustive Feature Selection (EFS)"

    print(f"\n--- Starting Feature Selection (Wrapper Method - {method_name}) ---")
    print(f"Model for EFS: {model_for_efs.__class__.__name__}")
    print(f"Scoring metric: {scoring_metric}")
    print(f"k_features range: {k_features_to_select}")

    if X_train_vetted.empty or y_train.empty:
        print("X_train or y_train is empty. Cannot perform feature selection.")
        return X_train_vetted, X_test_vetted, y_train, y_test, []

    # Initialize EFS (Exhaustive Feature Selector).
    # `min_features` and `max_features` define the range of subset sizes to explore.
    # `n_jobs=-1` uses all available CPU cores for parallel processing, speeding up the exhaustive search.
    efs = EFS(estimator=model_for_efs,
              min_features=k_features_to_select[0],
              max_features=k_features_to_select[1] if k_features_to_select[1] is not None else X_train_vetted.shape[1],
              scoring=scoring_metric,
              cv=cv,
              n_jobs=-1)

    # Fit EFS to the training data. This is the computationally intensive part where all subsets are evaluated.
    print(f"Fitting EFS on {X_train_vetted.shape[1]} vetted features and {X_train_vetted.shape[0]} samples...")
    efs.fit(X_train_vetted, y_train)
    print("EFS fitting completed.")

    # Get the best selected features based on the `scoring_metric`
    selected_feature_indices = list(efs.best_idx_) # Indices of the best features
    selected_feature_names = list(efs.best_feature_names_) # Names of the best features

    print(f"Number of features selected by {method_name}: {len(selected_feature_names)}")
    print(f"Selected features: {selected_feature_names}")

    # Create new DataFrames with only the selected features for both train and test sets
    X_train_final = X_train_vetted[selected_feature_names].copy()
    X_test_final = X_test_vetted[selected_feature_names].copy()

    # Save the final selected feature sets to CSV files
    X_train_final.to_csv(os.path.join(output_dir, 'X_train_final_selected_wrapper.csv'), index=False)
    X_test_final.to_csv(os.path.join(output_dir, 'X_test_final_selected_wrapper.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train_final_selected_wrapper.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test_final_selected_wrapper.csv'), index=False)

    # Save the list of selected feature names to a text file for easy reference
    with open(os.path.join(output_dir, 'final_selected_features_wrapper.txt'), 'w') as f:
        for feature in selected_feature_names:
            f.write(f"{feature}\n")
    print(f"Saved final selected features to '{output_dir}'")

    print("--- Feature Selection Completed ---")

    return X_train_final, X_test_final, y_train, y_test, selected_feature_names


def tune_random_forest_hyperparameters(X_train_data: pd.DataFrame, y_train_data: pd.Series,
                                       n_iter_search: int = 100, cv_folds: int = 10):
    """
    Performs hyperparameter tuning for RandomForestRegressor using RandomizedSearchCV.
    RandomizedSearchCV samples a fixed number of parameter settings from specified distributions.
    This is more efficient than GridSearchCV when the parameter space is large.

    Args:
        X_train_data (pd.DataFrame): The training features DataFrame.
        y_train_data (pd.Series): The training target Series.
        n_iter_search (int): Number of different parameter combinations to try.
        cv_folds (int): Number of cross-validation folds to use for evaluating each combination.

    Returns:
        tuple: (best_model, best_params)
               best_model (RandomForestRegressor): The best estimator found by RandomizedSearchCV.
               best_params (dict): Dictionary of the best hyperparameters found.
    """
    print(f"\n--- Starting Hyperparameter Tuning for RandomForestRegressor ---")
    print(f"Using RandomizedSearchCV with {n_iter_search} iterations and {cv_folds}-fold cross-validation.")

    # Define the parameter distributions.
    # RandomizedSearchCV samples values from these distributions.
    param_distributions = {
        'n_estimators': [100, 200, 300, 400, 500],  # Number of decision trees in the forest
        'max_features': ['sqrt', 'log2', 0.6, 0.8, 1.0], # Number of features to consider for best split
        'max_depth': [10, 20, 30, 40, 50, None],  # Max depth of each tree. 'None' means unlimited depth.
        'min_samples_split': [2, 5, 10],  # Min samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # Min samples required to be at a leaf node
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }

    # Initialize the RandomForestRegressor model with a random state for reproducibility
    rf_model = RandomForestRegressor(random_state=42)

    # Setup RandomizedSearchCV
    # `scoring='r2'` means we want to maximize the R-squared score (or minimize negative MSE if using that)
    # `n_jobs=-1` utilizes all available CPU cores for faster computation
    random_search = RandomizedSearchCV(estimator=rf_model,
                                       param_distributions=param_distributions,
                                       n_iter=n_iter_search,
                                       scoring='r2', # Optimizing for R-squared, higher is better
                                       cv=cv_folds,
                                       verbose=2, # Provides more detailed output during the search process
                                       random_state=42,
                                       n_jobs=-1)

    # Perform the search by fitting to the training data
    print("Performing RandomizedSearchCV fit...")
    random_search.fit(X_train_data, y_train_data)
    print("RandomizedSearchCV fit completed.")

    # Retrieve the best parameters and the best estimator (model) found
    best_params = random_search.best_params_
    best_model = random_search.best_estimator_
    best_score = random_search.best_score_ # This is the R-squared score

    print(f"Best hyperparameters found: {best_params}")
    print(f"Best cross-validation R-squared: {best_score:.4f}")
    print(f"--- Hyperparameter Tuning Completed ---")

    return best_model, best_params


def evaluate_baselines(X_train_all_scaled: pd.DataFrame, X_test_all_scaled: pd.DataFrame,
                       y_train_all: pd.Series, y_test_all: pd.Series):
    """
    Evaluates two baseline models to provide a reference point for the performance
    of our main machine learning model:
    1. Mean Baseline: Predicts the mean of the training target for all test samples.
    2. Hours of Sleep Baseline: Uses simple Linear Regression with 'sleep_hours' as the sole predictor.

    Args:
        X_train_all_scaled (pd.DataFrame): Scaled training features (should include 'sleep_hours' if available).
        X_test_all_scaled (pd.DataFrame): Scaled testing features (should include 'sleep_hours' if available).
        y_train_all (pd.Series): All training labels.
        y_test_all (pd.Series): All testing labels.

    Returns:
        dict: A dictionary containing MSE, RMSE, R2 for each baseline model.
    """
    print("\n--- Evaluating Baseline Models ---")
    baseline_metrics = {}

    # 1. Mean Baseline
    print("\n  - Mean Baseline:")
    mean_restfulness = y_train_all.mean() # Calculate the mean of the training target
    # Predict this mean for all samples in the test set
    y_pred_mean_baseline = np.full_like(y_test_all, mean_restfulness)

    # Calculate evaluation metrics for the Mean Baseline
    mse_mean_baseline = mean_squared_error(y_test_all, y_pred_mean_baseline)
    rmse_mean_baseline = np.sqrt(mse_mean_baseline)
    r2_mean_baseline = r2_score(y_test_all, y_pred_mean_baseline)

    print(f"    MSE: {mse_mean_baseline:.4f}")
    print(f"    RMSE: {rmse_mean_baseline:.4f}")
    print(f"    R-squared (R2): {r2_mean_baseline:.4f}")
    baseline_metrics['mean_baseline'] = {'MSE': mse_mean_baseline, 'RMSE': rmse_mean_baseline, 'R2': r2_mean_baseline}

    # 2. Hours of Sleep Baseline (Simple Linear Regression)
    # This baseline assesses if sleep duration alone can predict restfulness.
    print("\n  - Hours of Sleep Baseline (Linear Regression):")
    if 'sleep_hours' in X_train_all_scaled.columns and 'sleep_hours' in X_test_all_scaled.columns:
        # Extract only the 'sleep_hours' feature
        X_train_sleep_hours = X_train_all_scaled[['sleep_hours']]
        X_test_sleep_hours = X_test_all_scaled[['sleep_hours']]

        # IMPORTANT: Handle potential NaNs in 'sleep_hours' specifically for this baseline.
        # We only use rows where 'sleep_hours' is not NaN for training and testing this specific baseline.
        train_valid_mask = X_train_sleep_hours['sleep_hours'].notna()
        test_valid_mask = X_test_sleep_hours['sleep_hours'].notna()

        if train_valid_mask.sum() == 0 or test_valid_mask.sum() == 0:
            print("    Not enough valid 'sleep_hours' data for Linear Regression Baseline. Skipping.")
            baseline_metrics['sleep_hours_baseline'] = {'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
        else:
            lr_model = LinearRegression() # Initialize a simple Linear Regression model
            lr_model.fit(X_train_sleep_hours[train_valid_mask], y_train_all[train_valid_mask]) # Fit the model
            y_pred_sleep_hours_baseline = lr_model.predict(X_test_sleep_hours[test_valid_mask]) # Make predictions

            # Calculate evaluation metrics for the Linear Regression Baseline
            mse_sleep_hours_baseline = mean_squared_error(y_test_all[test_valid_mask], y_pred_sleep_hours_baseline)
            rmse_sleep_hours_baseline = np.sqrt(mse_sleep_hours_baseline)
            r2_sleep_hours_baseline = r2_score(y_test_all[test_valid_mask], y_pred_sleep_hours_baseline)

            print(f"    MSE: {mse_sleep_hours_baseline:.4f}")
            print(f"    RMSE: {rmse_sleep_hours_baseline:.4f}")
            print(f"    R-squared (R2): {r2_sleep_hours_baseline:.4f}")
            baseline_metrics['sleep_hours_baseline'] = {'MSE': mse_sleep_hours_baseline,
                                                        'RMSE': rmse_sleep_hours_baseline,
                                                        'R2': r2_sleep_hours_baseline}
    else:
        print("    'sleep_hours' column not found in data for Linear Regression Baseline. Skipping.")
        baseline_metrics['sleep_hours_baseline'] = {'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan}

    print("\n--- Baseline Evaluation Completed ---")
    return baseline_metrics


def evaluate_weekday_weekend_performance(X_test_final: pd.DataFrame, y_test_final: pd.Series,
                                         final_model: RandomForestRegressor, merged_data_full: pd.DataFrame):
    """
    Evaluates the final model's performance separately for weekdays and weekends.
    This helps understand if the model's predictive capability varies based on the day type.

    Args:
        X_test_final (pd.DataFrame): The final selected features for the test set (already scaled).
        y_test_final (pd.Series): The true labels for the test set.
        final_model (RandomForestRegressor): The trained final Random Forest model.
        merged_data_full (pd.DataFrame): The full merged dataset (used to retrieve original dates for test set mapping).
    """
    print("\n--- Evaluating Model Performance: Weekdays vs. Weekends ---")

    # IMPORTANT: We need to recreate the exact test set indices from the original `merged_data_full`
    # to correctly associate predictions with their original dates (and thus weekdays/weekends).
    # This relies on the `np.random.seed(42)` being consistent across `prepare_and_split_data` and this function.
    temp_uids = merged_data_full['uid'].unique()
    temp_test_data_collector = []

    for uid in temp_uids:
        uid_data = merged_data_full[merged_data_full['uid'] == uid].sort_values('date').reset_index(drop=True)
        features_uid = uid_data.drop(columns=['uid', 'date', 'restfulness'])
        target_uid = uid_data['restfulness']

        valid_indices = features_uid.notna().all(axis=1) # Same NaN filtering as in prepare_and_split_data
        features_uid = features_uid[valid_indices].reset_index(drop=True)
        target_uid = target_uid[valid_indices].reset_index(drop=True)

        if len(features_uid) < 2:
            continue

        train_size = int(0.8 * len(features_uid))
        if train_size == 0 or len(features_uid) - train_size == 0:
            continue

        # Re-generate the *same* random train/test split indices as done in `prepare_and_split_data`
        train_indices_temp = np.random.choice(len(features_uid), train_size, replace=False)
        test_indices_temp = np.setdiff1d(np.arange(len(features_uid)), train_indices_temp)

        # Select the original rows corresponding to the test set for this UID
        test_data_uid = uid_data.loc[valid_indices].iloc[test_indices_temp].copy()
        temp_test_data_collector.append(test_data_uid)

    if not temp_test_data_collector:
        print("  Insufficient data to perform Weekday/Weekend analysis.")
        return

    full_test_data_with_dates = pd.concat(temp_test_data_collector, ignore_index=True)
    full_test_data_with_dates['predicted'] = final_model.predict(X_test_final) # Add model predictions
    full_test_data_with_dates['actual'] = y_test_final.values # Add actual labels (ensure alignment with X_test_final)

    # Determine day of week: Monday=0, Sunday=6. Here, assuming Weekends are Friday (3), Saturday (4), Sunday (5).
    # Note: If local convention is different (e.g., Fri/Sat weekend), adjust these indices.
    full_test_data_with_dates['day_of_week'] = full_test_data_with_dates['date'].dt.dayofweek
    is_weekend_mask = (full_test_data_with_dates['day_of_week'] == 3) | \
                      (full_test_data_with_dates['day_of_week'] == 4) | \
                      (full_test_data_with_dates['day_of_week'] == 5)
    weekday_test_data = full_test_data_with_dates[~is_weekend_mask].copy()
    weekend_test_data = full_test_data_with_dates[is_weekend_mask].copy()

    # Print the dates classified as weekdays and weekends for verification
    print("\n  Dates considered Weekdays in Test Set (DayOfWeek: 0=Mon, 6=Sun):")
    if not weekday_test_data.empty:
        print(weekday_test_data[['date', 'day_of_week']].to_string(index=False))
    else:
        print("  No weekday dates in test set.")

    print("\n  Dates considered Weekends in Test Set (DayOfWeek: 0=Mon, 6=Sun):")
    if not weekend_test_data.empty:
        print(weekend_test_data[['date', 'day_of_week']].to_string(index=False))
    else:
        print("  No weekend dates in test set.")

    results = {}

    # Evaluate performance for Weekdays
    if not weekday_test_data.empty:
        mse_weekday = mean_squared_error(weekday_test_data['actual'], weekday_test_data['predicted'])
        rmse_weekday = np.sqrt(mse_weekday)
        r2_weekday = r2_score(weekday_test_data['actual'], weekday_test_data['predicted'])
        results['weekday'] = {'MSE': mse_weekday, 'RMSE': rmse_weekday, 'R2': r2_weekday}
    else:
        results['weekday'] = {'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
        print("  No weekday data in test set for analysis.")

    # Evaluate performance for Weekends
    if not weekend_test_data.empty:
        mse_weekend = mean_squared_error(weekend_test_data['actual'], weekend_test_data['predicted'])
        rmse_weekend = np.sqrt(mse_weekend)
        r2_weekend = r2_score(weekend_test_data['actual'], weekend_test_data['predicted'])
        results['weekend'] = {'MSE': mse_weekend, 'RMSE': rmse_weekend, 'R2': r2_weekend}
    else:
        results['weekend'] = {'MSE': np.nan, 'RMSE': np.nan, 'R2': np.nan}
        print("  No weekend data in test set for analysis.")

    print("\n--- Weekday vs. Weekend Performance Summary ---")
    print(f"{'Day Type':<15}{'MSE':<10}{'RMSE':<10}{'R2':<10}")
    print("-" * 45)
    print(
        f"{'Weekday':<15}{results['weekday']['MSE']:.4f}{results['weekday']['RMSE']:.4f}{results['weekday']['R2']:.4f}")
    print(
        f"{'Weekend':<15}{results['weekend']['MSE']:.4f}{results['weekend']['RMSE']:.4f}{results['weekend']['R2']:.4f}")
    print("-" * 45)
    print("--- Weekday/Weekend Analysis Completed ---")


def plot_model_performance(y_test_actual: pd.Series, y_pred_predicted: np.ndarray,
                           model: RandomForestRegressor, feature_names: list,
                           title_prefix: str = "Model"):
    """
    Generates visualizations to assess and communicate model performance.
    Includes:
    1. Prediction vs. Actual Plot: Shows how well predicted values align with true values.
    2. Feature Importance Plot: Ranks features by their contribution to the model's predictions (for RandomForest).
    3. Error Distribution Plots: Visualizes the distribution of prediction errors.

    Args:
        y_test_actual (pd.Series): The true labels from the test set.
        y_pred_predicted (np.ndarray): The predicted values from the model on the test set.
        model (RandomForestRegressor): The trained Random Forest model.
        feature_names (list): A list of feature names used by the model.
        title_prefix (str): A prefix for plot titles (e.g., "Final Model").
    """
    print(f"\n--- Generating Visualizations for {title_prefix} Performance ---")

    # Round and clip predictions to match the 1-10 integer scale of actual restfulness ratings for better visualization
    y_pred_rounded_clipped = np.clip(np.round(y_pred_predicted), 1, 10)

    # 1. Prediction vs. Actual Plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_test_actual, y=y_pred_rounded_clipped, alpha=0.6, color='blue')
    # Plot a diagonal line (y=x) representing perfect predictions
    plt.plot([min(y_test_actual.min(), y_pred_rounded_clipped.min()),
              max(y_test_actual.max(), y_pred_rounded_clipped.max())],
             [min(y_test_actual.min(), y_pred_rounded_clipped.min()),
              max(y_test_actual.max(), y_pred_rounded_clipped.max())],
             color='red', linestyle='--', label='Ideal Prediction (y=x)')
    plt.title(f'{title_prefix}: Prediction vs. Actual')
    plt.xlabel('Actual Restfulness Rating')
    plt.ylabel('Predicted Restfulness Rating')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Feature Importance Analysis (applicable for tree-based models like RandomForest)
    if hasattr(model, 'feature_importances_') and feature_names:
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False) # Sort features by their importance score

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'{title_prefix}: Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Feature importance not available for {title_prefix} model or feature names are missing.")

    # 3. Error Distribution
    # Calculate the prediction errors (actual - predicted)
    errors = y_test_actual - y_pred_predicted

    plt.figure(figsize=(10, 5))

    # Histogram of errors: shows the frequency distribution of errors.
    # A distribution centered around zero indicates unbiased predictions.
    plt.subplot(1, 2, 1)
    sns.histplot(errors, kde=True, bins=15, color='purple', edgecolor='black')
    plt.title(f'{title_prefix}: Error Distribution (Histogram)')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)

    # Box plot of errors: provides a visual summary of the error distribution,
    # including median, quartiles, and outliers.
    plt.subplot(1, 2, 2)
    sns.boxplot(y=errors, color='lightgreen')
    plt.title(f'{title_prefix}: Error Distribution (Box Plot)')
    plt.ylabel('Error (Actual - Predicted)')
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.show()

    print(f"Visualizations for {title_prefix} performance displayed.")


if __name__ == "__main__":
    # Define the project directory where all data and output will be managed
    project_dir = r"C:\Users\User\Documents\05"
    output_dir_main = os.path.join(project_dir, 'sensor_data')

    try:
        print("--- Starting Data Processing Pipeline ---")

        # Ensure the project directory exists
        if not os.path.exists(project_dir):
            raise FileNotFoundError(f"Directory {project_dir} does not exist. Please check the path.")

        # Step 1-5: Data Loading, Cleaning, Feature Engineering, and Merging
        print("\n--- Phase 1: Data Loading, Cleaning & Feature Engineering ---")
        sensor_dfs, _ = load_sensor_data(project_dir) # Load raw sensor data
        quest_df_sleep_hours = load_questionnaire_data(project_dir) # Load questionnaire data for sleep hours
        sensor_dfs = handle_missing_values(sensor_dfs, output_dir_main) # Impute missing values in sensor data
        feature_dfs = compute_features(sensor_dfs, quest_df_sleep_hours) # Compute features from sensor data
        all_features_df = merge_features(feature_dfs, quest_df_sleep_hours, output_dir_main) # Merge all features

        # Reload the merged data including 'restfulness' label. This step reads the `all_features.csv`
        # created by `merge_features` and combines it with the restfulness labels from questionnaires.
        merged_data_full, _ = load_data(project_dir)
        print("Data loading, cleaning, and feature engineering completed.")

        # Step 6: Prepare and Split Data (Includes Individual Normalization)
        # This function performs the critical train/test split on a per-UID basis and scales features.
        print("\n--- Phase 2: Prepare and Split Data (Individual Normalization) ---")
        split_results = prepare_and_split_data(
            merged_data_full, output_dir_main
        )

        if split_results is None:
            print("Failed to prepare and split data due to insufficient valid data. Exiting process.")
        else:
            X_train_all_scaled, X_test_all_scaled, y_train_all, y_test_all = split_results
            print("Train and Test preparation completed successfully.")

            # --- Evaluation after Phase 2 (Initial model on all scaled features) ---
            # This helps understand the model performance before vetting and specific feature selection.
            print("\n--- Evaluation after Phase 2 (all features, scaled) ---")
            model_after_prep = RandomForestRegressor(random_state=42)
            model_after_prep.fit(X_train_all_scaled, y_train_all)
            y_pred_after_prep = model_after_prep.predict(X_test_all_scaled)

            mse_after_prep = mean_squared_error(y_test_all, y_pred_after_prep)
            rmse_after_prep = np.sqrt(mse_after_prep) # Calculate RMSE

            print(f"Model Performance (all {X_train_all_scaled.shape[1]} features, scaled):")
            print(f"  Mean Squared Error (MSE): {mse_after_prep:.4f}")
            print(f"  Root Mean Squared Error (RMSE): {rmse_after_prep:.4f}")

            # Step 7: Perform Vetting (Outlier capping, MI-based selection, redundancy removal, ReliefF)
            # This step reduces the number of features to a more manageable and relevant set (e.g., 15 features).
            print("\n--- Phase 3: Vetting ---")
            X_train_vetted, X_test_vetted, y_train_vetted, y_test_vetted, final_selected_features_vetting = vetting(
                X_train_all_scaled, X_test_all_scaled, y_train_all, y_test_all,
                target_column_name='restfulness',
                num_features_to_keep=15, # Target number of features after vetting
                mi_threshold_for_initial_selection=0.001, # Threshold for initial MI selection
                mi_threshold_for_redundancy=0.7 # Threshold for detecting redundant features
            )
            print("Vetting completed.")
            print(f"Number of features after Vetting: {len(final_selected_features_vetting)}")

            # --- Evaluation after Phase 3 (Model on vetted features) ---
            # Evaluate model performance on the features after vetting.
            print("\n--- Evaluation after Phase 3 (Vetted features) ---")
            model_after_vetting = RandomForestRegressor(random_state=42)
            model_after_vetting.fit(X_train_vetted, y_train_vetted) # Train model on vetted features
            y_pred_after_vetting = model_after_vetting.predict(X_test_vetted)

            mse_after_vetting = mean_squared_error(y_test_vetted, y_pred_after_vetting)
            rmse_after_vetting = np.sqrt(mse_after_vetting)

            print(f"Model Performance (after Vetting - {X_train_vetted.shape[1]} features):")
            print(f"  Mean Squared Error (MSE): {mse_after_vetting:.4f}")
            print(f"  Root Mean Squared Error (RMSE): {rmse_after_vetting:.4f}")

            # Step 8: Perform Feature Selection using Wrapper Method (Exhaustive Feature Selector - EFS)
            # This step finds the optimal subset of features from the vetted set (e.g., exactly 10 features).
            print("\n--- Phase 4: Wrapper Feature Selection (Using EFS) ---")
            X_train_final, X_test_final, y_train_final, y_test_final, final_selected_features_wrapper = \
                feature_selection_wrapper(
                    X_train_vetted, X_test_vetted, y_train_vetted, y_test_vetted, output_dir_main,
                    model_for_efs=RandomForestRegressor(random_state=42, n_estimators=50, max_depth=8),
                    scoring_metric='neg_mean_squared_error',
                    k_features_to_select=(10, 10), # Select exactly 10 features
                    cv=5)

            print("Wrapper Feature Selection completed.")
            print(f"Number of features selected by Wrapper method: {len(final_selected_features_wrapper)}")

            # --- Phase 5: Hyperparameter Tuning for Final Model ---
            # Optimize the RandomForestRegressor's parameters on the finally selected features.
            print("\n--- Phase 5: Hyperparameter Tuning for Final Model ---")
            final_model, best_hyperparameters = tune_random_forest_hyperparameters(
                X_train_final, y_train_final,
                n_iter_search=100, # Number of parameter settings to try
                cv_folds=10 # Cross-validation folds for tuning
            )
            print("Hyperparameter tuning completed. Best parameters applied to final model.")

            # Calculate predictions for the final, tuned model
            y_pred_final = final_model.predict(X_test_final)

            # Evaluate the final model's quantitative metrics
            mse_final = mean_squared_error(y_test_final, y_pred_final)
            r2_final = r2_score(y_test_final, y_pred_final)
            rmse_final = np.sqrt(mse_final)
            # Round and clip predictions for presentation and consistency with the 1-10 rating scale
            y_pred_final_rounded = np.clip(np.round(y_pred_final), 1, 10)

            print(f"Final Model Performance on Test Set (after Wrapper FS & Hyperparameter Tuning):")
            print(f"Mean Squared Error (MSE): {mse_final:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse_final:.4f}")
            print(f"R-squared (R2): {r2_final:.4f}")

            # Evaluate and Compare Baselines against the final model
            baseline_results = evaluate_baselines(X_train_all_scaled, X_test_all_scaled, y_train_all, y_test_all)

            print("\n--- Comparative Model Performance ---")
            print(f"{'Model':<30}{'MSE':<10}{'RMSE':<10}{'R2':<10}")
            print("-" * 60)
            print(f"{'Random Forest (Final)':<30}{mse_final:<10.4f}{rmse_final:<10.4f}{r2_final:<10.4f}")
            print(
                f"{'Mean Baseline':<30}{baseline_results['mean_baseline']['MSE']:<10.4f}{baseline_results['mean_baseline']['RMSE']:<10.4f}{baseline_results['mean_baseline']['R2']:<10.4f}")
            if 'sleep_hours_baseline' in baseline_results and not np.isnan(baseline_results['sleep_hours_baseline']['MSE']):
                print(
                    f"{'Sleep Hours Baseline (LR)':<30}{baseline_results['sleep_hours_baseline']['MSE']:<10.4f}{baseline_results['sleep_hours_baseline']['RMSE']:<10.4f}{baseline_results['sleep_hours_baseline']['R2']:<10.4f}")
            else:
                print(f"{'Sleep Hours Baseline (LR)':<30}{'N/A':<10}{'N/A':<10}{'N/A':<10}")
            print("-" * 60)

            # Generate and display visualizations for the final model
            plot_model_performance(y_test_final, y_pred_final, final_model, final_selected_features_wrapper,
                                   "Final Random Forest Model")

            # Evaluate Weekday vs. Weekend performance for the final model
            evaluate_weekday_weekend_performance(X_test_final, y_test_final, final_model, merged_data_full)

            # Print a small sample of actual vs. predicted values for qualitative assessment
            print(
                f"\nActual vs Predicted for a sample of final test data (using {len(final_selected_features_wrapper)} features):")
            num_samples_to_print = min(10, len(y_test_final))
            for i in range(num_samples_to_print):
                print(f"Actual: {y_test_final.iloc[i]}, Predicted: {y_pred_final_rounded[i]}")

            print(f"\n--- Pipeline Completed ---")
            print(f"Final vetted and selected matrices saved to {output_dir_main}")

    except Exception as e:
        print(f"An error occurred during the processing pipeline: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging