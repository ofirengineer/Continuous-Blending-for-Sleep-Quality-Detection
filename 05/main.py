import pandas as np
import json
import math
from datetime import datetime, timedelta
import os


# Custom haversine function to calculate distance
def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    lat1, lon1, lat2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r


# Load sensor data from sensor_data directory
def load_sensor_data(project_dir):
    sensor_dir = os.path.join(project_dir, 'sensor_data')
    output_dir = sensor_dir
    os.makedirs(output_dir, exist_ok=True)

    sensor_dfs = {}
    sensors = ['accelerometer', 'calls', 'light', 'screen', 'location', 'wifi']
    file_names = {
        'accelerometer': 'accelerometer_data.csv',
        'calls': 'calls_data.csv',
        'light': 'light_data.csv',
        'screen': 'screen_data.csv',
        'location': 'location_data.csv',
        'wifi': 'wifi_data.csv'
    }

    for sensor in sensors:
        file_path = os.path.join(sensor_dir, file_names[sensor])
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df = df.dropna(subset=['uid', 'datetime'])
            sensor_dfs[sensor] = df
            print(f"Loaded {sensor} data from '{file_path}'")
        except FileNotFoundError:
            print(f"File {file_path} not found. Skipping {sensor}.")
            sensor_dfs[sensor] = pd.DataFrame()

    return sensor_dfs, output_dir


# Load and merge questionnaire data
def load_questionnaire_data(project_dir):
    files = ['Session_A_Label.csv', 'Session_B_Label.csv', 'Session_C_Label.csv']
    dfs = []

    for file in files:
        file_path = os.path.join(project_dir, file)
        try:
            try:
                df = pd.read_csv(file_path, encoding='latin1')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='Windows-1252')

            print(f"Columns in {file}: {df.columns.tolist()}")

            sleep_col = next((col for col in df.columns if col.startswith("How long did you sleep last night")), None)
            if sleep_col is None:
                raise KeyError(f"No column starting with 'How long did you sleep last night' found in {file}")

            dfs.append(df[['Timestamp', 'uid', sleep_col]])
        except FileNotFoundError:
            print(f"{file_path} not found.")
            raise
        except Exception as e:
            print(f"Error reading {file}: {e}")
            raise

    quest_df = pd.concat(dfs, ignore_index=True)
    quest_df['Timestamp'] = pd.to_datetime(quest_df['Timestamp'], errors='coerce', dayfirst=True)

    def convert_time_to_hours(time_str):
        if pd.isna(time_str):
            return np.nan
        try:
            h, m, s = map(int, time_str.split(':'))
            return h + m / 60 + s / 3600
        except:
            return np.nan

    quest_df['sleep_hours'] = quest_df.iloc[:, -1].apply(convert_time_to_hours)
    return quest_df


# Handle missing values smartly
def handle_missing_values(sensor_dfs, output_dir):
    for sensor in sensor_dfs:
        if sensor_dfs[sensor].empty:
            print(f"Skipping empty {sensor} data.")
            continue

        sensor_data = sensor_dfs[sensor].copy()

        if sensor == 'accelerometer':
            for col in ['x', 'y', 'z']:
                sensor_data[col] = pd.to_numeric(sensor_data[col], errors='coerce')
                night_mask = sensor_data['datetime'].dt.hour.between(0, 6)
                sensor_data.loc[night_mask & sensor_data[col].isna(), col] = 9.81 if col == 'z' else 0
                sensor_data[col] = sensor_data.groupby('uid')[col].ffill()
                for uid in sensor_data['uid'].unique():
                    uid_mask = sensor_data['uid'] == uid
                    for idx in sensor_data[uid_mask & sensor_data[col].isna()].index:
                        time = sensor_data.loc[idx, 'datetime']
                        window = sensor_data[uid_mask & (
                            sensor_data['datetime'].between(time - timedelta(hours=1), time + timedelta(hours=1)))]
                        if not window[col].isna().all():
                            sensor_data.loc[idx, col] = window[col].mean()
                mean_values = sensor_data.groupby('uid')[col].mean()
                sensor_data[col] = sensor_data[col].fillna(sensor_data['uid'].map(mean_values))
                sensor_data[col] = sensor_data[col].fillna(0)

        elif sensor == 'calls':
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce')
            sensor_data['value'] = sensor_data['value'].fillna(0)
            sensor_data['sensor_status'] = sensor_data.apply(
                lambda row: 2 if pd.isna(row['sensor_status']) and 'data' in row and isinstance(row['data'],
                                                                                                str) and '"MISSED"' in
                                 row['data'] else row['sensor_status'], axis=1)
            sensor_data['sensor_status'] = sensor_data['sensor_status'].replace({
                'OUTGOING': 0, 'INCOMING': 1, 'REJECTED': 2
            })
            sensor_data['sensor_status'] = pd.to_numeric(sensor_data['sensor_status'], errors='coerce')
            sensor_data['sensor_status'] = sensor_data['sensor_status'].fillna(2)

        elif sensor == 'light':
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce')
            sensor_data.loc[sensor_data['value'] > 1e6, 'value'] = 0
            night_mask = sensor_data['datetime'].dt.hour.between(0, 6)
            sensor_data.loc[night_mask, 'value'] = sensor_data.loc[night_mask, 'value'].fillna(0)
            for uid in sensor_data['uid'].unique():
                uid_mask = sensor_data['uid'] == uid
                for idx in sensor_data[uid_mask & sensor_data['value'].isna()].index:
                    time = sensor_data.loc[idx, 'datetime']
                    window = sensor_data[uid_mask & (
                        sensor_data['datetime'].between(time - timedelta(hours=1), time + timedelta(hours=1)))]
                    if not window['value'].isna().all():
                        sensor_data.loc[idx, 'value'] = window['value'].mean()
            mean_values = sensor_data.groupby('uid')['value'].mean()
            sensor_data['value'] = sensor_data['value'].fillna(sensor_data['uid'].map(mean_values))
            sensor_data['value'] = sensor_data['value'].fillna(10)

        elif sensor == 'screen':
            sensor_data['value'] = sensor_data['data'].apply(
                lambda x: 1 if isinstance(x, str) and '"on"' in x else 0 if isinstance(x,
                                                                                       str) and '"off"' in x else np.nan)
            sensor_data['value'] = pd.to_numeric(sensor_data['value'], errors='coerce')
            night_mask = sensor_data['datetime'].dt.hour.between(0, 6)
            sensor_data.loc[night_mask, 'value'] = sensor_data.loc[night_mask, 'value'].fillna(0)
            sensor_data['value'] = sensor_data.groupby('uid')['value'].ffill()
            sensor_data['value'] = sensor_data['value'].fillna(0)

        elif sensor == 'location':
            sensor_data['suuid'] = sensor_data['suuid'].fillna('unknown')
            for col in ['x', 'y']:
                sensor_data[col] = pd.to_numeric(sensor_data[col], errors='coerce')
                sensor_data[col] = sensor_data.groupby('uid')[col].ffill()
                for uid in sensor_data['uid'].unique():
                    uid_mask = sensor_data['uid'] == uid
                    for idx in sensor_data[uid_mask & sensor_data[col].isna()].index:
                        time = sensor_data.loc[idx, 'datetime']
                        window = sensor_data[uid_mask & (
                            sensor_data['datetime'].between(time - timedelta(hours=1), time + timedelta(hours=1)))]
                        if not window[col].isna().all():
                            sensor_data.loc[idx, col] = window[col].mean()
                mean_values = sensor_data.groupby('uid')[col].mean()
                sensor_data[col] = sensor_data[col].fillna(sensor_data['uid'].map(mean_values))
                sensor_data[col] = sensor_data[col].fillna(0)
            sensor_data['sensor_status'] = sensor_data['sensor_status'].replace({'ON': 1, 'OFF': 0})
            sensor_data['sensor_status'] = pd.to_numeric(sensor_data['sensor_status'], errors='coerce')
            sensor_data['sensor_status'] = sensor_data['sensor_status'].fillna(0)

        elif sensor == 'wifi':
            sensor_data['suuid'] = sensor_data['suuid'].fillna('NULL')
            sensor_data['level'] = pd.to_numeric(sensor_data['level'], errors='coerce')
            for uid in sensor_data['uid'].unique():
                uid_mask = sensor_data['uid'] == uid
                for idx in sensor_data[uid_mask & sensor_data['level'].isna()].index:
                    time = sensor_data.loc[idx, 'datetime']
                    window = sensor_data[uid_mask & (
                        sensor_data['datetime'].between(time - timedelta(hours=1), time + timedelta(hours=1)))]
                    if not window['level'].isna().all():
                        sensor_data.loc[idx, 'level'] = window['level'].mean()
            mean_values = sensor_data.groupby('uid')['level'].mean()
            sensor_data['level'] = sensor_data['level'].fillna(sensor_data['uid'].map(mean_values))
            sensor_data['level'] = sensor_data['level'].fillna(-100)

        output_path = os.path.join(output_dir, f'{sensor}_data_clean.csv')
        sensor_data.to_csv(output_path, index=False)
        print(f"Saved cleaned {sensor} data to '{output_path}'")
        sensor_dfs[sensor] = sensor_data

    return sensor_dfs


# Compute features for each sensor
def compute_features(sensor_dfs, quest_df):
    feature_dfs = {}

    for sensor in sensor_dfs:
        if sensor_dfs[sensor].empty:
            print(f"Skipping feature computation for empty {sensor} data.")
            feature_dfs[sensor] = pd.DataFrame()
            continue

        sensor_data = sensor_dfs[sensor].copy()
        sensor_data['date'] = sensor_data['datetime'].dt.date
        sensor_data['hour'] = sensor_data['datetime'].dt.hour

        if sensor == 'accelerometer':
            # Basic features
            sensor_data['magnitude'] = np.sqrt(sensor_data['x'] ** 2 + sensor_data['y'] ** 2 + sensor_data['z'] ** 2)
            # Night (22:00–06:30) features
            night_mask = (sensor_data['hour'] >= 22) | (sensor_data['hour'] <= 6)
            # Energy
            sensor_data['energy'] = sensor_data['magnitude'] ** 2
            # Movement changes
            sensor_data['is_moving'] = sensor_data['magnitude'].apply(lambda x: 1 if abs(x - 9.81) > 1 else 0)
            sensor_data['movement_change'] = sensor_data.groupby('uid')['is_moving'].diff().abs().fillna(0)
            # Significant movements
            sensor_data['significant_movement'] = sensor_data['magnitude'].apply(lambda x: 1 if x > 10 else 0)

            features = sensor_data.groupby(['uid', 'date']).agg({
                'magnitude': ['mean', 'std', 'min', 'max'],
                'x': ['mean', 'std'],
                'y': ['mean', 'std'],
                'z': ['mean', 'std'],
                'energy': lambda x: x[night_mask].sum(),  # Night energy
                'movement_change': lambda x: x[night_mask].sum(),  # Night movement changes
                'significant_movement': lambda x: x[night_mask].sum()  # Significant movements
            }).reset_index()

            # Rest periods (approximated as consecutive static periods)
            rest_periods = []
            for uid in sensor_data['uid'].unique():
                uid_data = sensor_data[(sensor_data['uid'] == uid) & night_mask].sort_values('datetime')
                uid_data['is_static'] = uid_data['magnitude'].apply(lambda x: abs(x - 9.81) < 0.5)
                uid_data['group'] = (uid_data['is_static'] != uid_data['is_static'].shift()).cumsum()
                static_groups = uid_data[uid_data['is_static']].groupby(['date', 'group'])['datetime'].agg(
                    ['min', 'max'])
                static_groups['duration'] = (static_groups['max'] - static_groups['min']).dt.total_seconds() / 3600
                rest = static_groups.groupby('date')['duration'].sum().reset_index()
                rest['uid'] = uid
                rest_periods.append(rest)

            rest_df = pd.concat(rest_periods).groupby(['uid', 'date'])['duration'].sum().reset_index()
            rest_df.columns = ['uid', 'date', 'rest_duration']

            features.columns = ['uid', 'date', 'acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_min',
                                'acc_magnitude_max',
                                'acc_x_mean', 'acc_x_std', 'acc_y_mean', 'acc_y_std', 'acc_z_mean', 'acc_z_std',
                                'acc_night_energy', 'acc_night_movement_changes', 'acc_night_significant_movements']
            features = pd.merge(features, rest_df, on=['uid', 'date'], how='left')
            features['acc_rest_duration'] = features['rest_duration'].fillna(0)
            features = features.drop(columns=['rest_duration'])

        elif sensor == 'calls':
            # Late calls (after 21:00)
            late_mask = sensor_data['hour'] >= 21
            # Last call time to sleep window (23:00)
            sensor_data['time_to_sleep'] = sensor_data['datetime'].apply(
                lambda x: (datetime(x.year, x.month, x.day, 23,
                                    0) - x).total_seconds() / 3600 if x.hour < 23 else np.nan)

            features = sensor_data.groupby(['uid', 'date']).agg({
                'value': ['sum', 'count'],
                'sensor_status': lambda x: (x == 1).sum(),
                'value': lambda x: x[late_mask].sum(),  # Late call duration
                'sensor_status': lambda x: x[late_mask].count(),  # Late call count
                'time_to_sleep': 'min'  # Min time to sleep window
            }).reset_index()

            features.columns = ['uid', 'date', 'call_duration_sum', 'call_count', 'incoming_call_count',
                                'late_call_duration', 'late_call_count', 'last_call_to_sleep']
            features['last_call_to_sleep'] = features['last_call_to_sleep'].fillna(0)  # No calls = 0

        elif sensor == 'light':
            # Night (20:00–06:30) features
            night_mask = (sensor_data['hour'] >= 20) | (sensor_data['hour'] <= 6)
            # High light exposure
            sensor_data['high_light'] = sensor_data['value'].apply(lambda x: 1 if x > 100 else 0)
            # Low light time
            sensor_data['low_light'] = sensor_data['value'].apply(lambda x: 1 if x < 10 else 0)

            features = sensor_data.groupby(['uid', 'date']).agg({
                'value': ['mean', 'std', 'max'],
                'value': lambda x: x[night_mask].mean(),  # Night light mean
                'value': lambda x: x[night_mask].std(),  # Night light std
                'high_light': lambda x: x[night_mask].sum(),  # High light exposure
                'low_light': lambda x: x[night_mask].mean()  # Proportion of low light
            }).reset_index()

            features.columns = ['uid', 'date', 'light_mean', 'light_std', 'light_max',
                                'light_night_mean', 'light_night_std', 'light_night_high_exposure',
                                'light_night_low_proportion']

        elif sensor == 'screen':
            # Night (22:00–06:30) and pre-sleep (20:00–22:00)
            night_mask = (sensor_data['hour'] >= 22) | (sensor_data['hour'] <= 6)
            pre_sleep_mask = (sensor_data['hour'] >= 20) & (sensor_data['hour'] < 22)
            day_mask = (sensor_data['hour'] >= 6) & (sensor_data['hour'] < 22)
            # Screen activations
            sensor_data['screen_change'] = sensor_data.groupby('uid')['value'].diff().eq(1).astype(int)
            # Last screen use
            sensor_data['time_to_sleep'] = sensor_data['datetime'].apply(
                lambda x: (datetime(x.year, x.month, x.day, 23, 0) - x).total_seconds() / 3600 if x.hour < 23 and
                                                                                                  sensor_data.loc[
                                                                                                      x.name, 'value'] == 1 else np.nan)

            features = sensor_data.groupby(['uid', 'date']).agg({
                'value': ['sum', 'count'],
                'value': lambda x: x[pre_sleep_mask].sum(),  # Pre-sleep screen time
                'screen_change': lambda x: x[night_mask].sum(),  # Night activations
                'time_to_sleep': 'min',  # Last screen use
                'value': lambda x: x[night_mask].sum() / (x[day_mask].sum() + 1e-10)  # Night/day ratio
            }).reset_index()

            features.columns = ['uid', 'date', 'screen_on_sum', 'screen_event_count',
                                'screen_pre_sleep_sum', 'screen_night_activations',
                                'screen_last_use_to_sleep', 'screen_night_day_ratio']
            features['screen_last_use_to_sleep'] = features['screen_last_use_to_sleep'].fillna(0)

        elif sensor == 'location':
            # Night (22:00–06:30)
            night_mask = (sensor_data['hour'] >= 22) | (sensor_data['hour'] <= 6)
            pre_sleep_mask = (sensor_data['hour'] >= 20) & (sensor_data['hour'] < 22)
            # Estimate home location (mean x, y at night)
            home_loc = sensor_data[night_mask].groupby('uid')[['x', 'y']].mean().reset_index()
            home_loc.columns = ['uid', 'home_x', 'home_y']
            sensor_data = pd.merge(sensor_data, home_loc, on='uid', how='left')
            sensor_data['away_from_home'] = sensor_data.apply(
                lambda row: haversine((row['x'], row['y']), (row['home_x'], row['home_y'])) > 0.1, axis=1)
            # Distance and speed
            sensor_data = sensor_data.sort_values(['uid', 'datetime'])
            sensor_data['prev_x'] = sensor_data.groupby('uid')['x'].shift(1)
            sensor_data['prev_y'] = sensor_data.groupby('uid')['y'].shift(1)
            sensor_data['distance'] = sensor_data.apply(
                lambda row: haversine((row['x'], row['y']), (row['prev_x'], row['prev_y'])) if not pd.isna(
                    row['prev_x']) else 0, axis=1)
            sensor_data['time_diff'] = sensor_data.groupby('uid')['datetime'].diff().dt.total_seconds() / 3600
            sensor_data['speed'] = sensor_data['distance'] / (sensor_data['time_diff'] + 1e-10)
            # Unique locations
            sensor_data['location_cluster'] = sensor_data.apply(
                lambda row: f"{round(row['x'], 2)}_{round(row['y'], 2)}", axis=1)

            features = sensor_data.groupby(['uid', 'date']).agg({
                'distance': ['sum', 'mean'],
                'x': ['mean', 'std'],
                'y': ['mean', 'std'],
                'away_from_home': lambda x: x[night_mask].sum(),  # Time away from home
                'speed': lambda x: x[pre_sleep_mask].mean(),  # Pre-sleep speed
                'location_cluster': 'nunique'  # Unique locations
            }).reset_index()

            features.columns = ['uid', 'date', 'distance_sum', 'distance_mean', 'lat_mean', 'lat_std', 'lon_mean',
                                'lon_std',
                                'loc_night_away_from_home', 'loc_pre_sleep_speed', 'loc_unique_count']

        elif sensor == 'wifi':
            # Night (22:00–06:30)
            night_mask = (sensor_data['hour'] >= 22) | (sensor_data['hour'] <= 6)
            # Home WiFi (most common suuid at night)
            home_wifi = sensor_data[night_mask].groupby('uid')['suuid'].agg(
                lambda x: x.mode()[0] if not x.mode().empty else 'NULL').reset_index()
            home_wifi.columns = ['uid', 'home_suuid']
            sensor_data = pd.merge(sensor_data, home_wifi, on='uid', how='left')
            sensor_data['is_home_wifi'] = sensor_data['suuid'] == sensor_data['home_suuid']
            # WiFi changes
            sensor_data['wifi_change'] = sensor_data.groupby('uid')['suuid'].diff().ne(0).astype(int)

            features = sensor_data.groupby(['uid', 'date']).agg({
                'level': ['mean', 'std'],
                'suuid': 'nunique',
                'wifi_change': lambda x: x[night_mask].sum(),  # Night WiFi changes
                'level': lambda x: x[night_mask].mean(),  # Night signal strength
                'is_home_wifi': lambda x: x[night_mask].sum()  # Time on home WiFi
            }).reset_index()

            features.columns = ['uid', 'date', 'wifi_level_mean', 'wifi_level_std', 'wifi_unique_count',
                                'wifi_night_changes', 'wifi_night_signal_mean', 'wifi_night_home_time']

        feature_dfs[sensor] = features

    # Cross-sensor features
    cross_features = []
    for uid in sensor_dfs['accelerometer']['uid'].unique():
        for date in sensor_dfs['accelerometer']['date'].unique():
            acc_data = sensor_dfs['accelerometer'][
                (sensor_dfs['accelerometer']['uid'] == uid) & (sensor_dfs['accelerometer']['date'] == date)]
            screen_data = sensor_dfs['screen'][
                (sensor_dfs['screen']['uid'] == uid) & (sensor_dfs['screen']['date'] == date)]
            light_data = sensor_dfs['light'][
                (sensor_dfs['light']['uid'] == uid) & (sensor_dfs['light']['date'] == date)]

            pre_sleep_mask = (acc_data['hour'] >= 20) & (acc_data['hour'] < 22)
            night_mask = (acc_data['hour'] >= 22) | (acc_data['hour'] <= 6)

            # Pre-sleep activity
            acc_energy = acc_data[pre_sleep_mask]['magnitude'].apply(lambda x: x ** 2).sum() if not acc_data[
                pre_sleep_mask].empty else 0
            screen_time = screen_data[pre_sleep_mask]['value'].sum() if not screen_data[pre_sleep_mask].empty else 0
            light_level = light_data[pre_sleep_mask]['value'].mean() if not light_data[pre_sleep_mask].empty else 0
            activity_score = (acc_energy / (acc_energy.max() + 1e-10)) + screen_time + (light_level / 100)

            # Night disturbances
            acc_disturb = acc_data[night_mask]['magnitude'].apply(lambda x: 1 if abs(x - 9.81) > 1 else 0).sum() if not \
            acc_data[night_mask].empty else 0
            screen_disturb = screen_data[night_mask]['value'].diff().eq(1).sum() if not screen_data[
                night_mask].empty else 0
            light_disturb = light_data[night_mask]['value'].apply(lambda x: 1 if x > 100 else 0).sum() if not \
            light_data[night_mask].empty else 0
            disturbances = acc_disturb + screen_disturb + light_disturb

            cross_features.append({
                'uid': uid,
                'date': date,
                'pre_sleep_activity_score': activity_score,
                'night_disturbances': disturbances
            })

    cross_df = pd.DataFrame(cross_features)
    feature_dfs['cross_sensor'] = cross_df

    return feature_dfs


# Merge features with questionnaire data
def merge_features(feature_dfs, quest_df, output_dir):
    merged_dfs = {}

    for sensor in feature_dfs:
        if feature_dfs[sensor].empty:
            print(f"Skipping merge for empty {sensor} features.")
            merged_dfs[sensor] = pd.DataFrame()
            continue

        features = feature_dfs[sensor].copy()
        features['date'] = pd.to_datetime(features['date'])
        quest_df['date'] = quest_df['Timestamp'].dt.date
        quest_df['date'] = pd.to_datetime(quest_df['date'])

        merged = pd.merge(features, quest_df[['uid', 'date', 'sleep_hours']], on=['uid', 'date'], how='left')
        merged['sleep_hours'] = merged['sleep_hours'].fillna(quest_df['sleep_hours'].median())

        output_path = os.path.join(output_dir, f'{sensor}_features.csv')
        merged.to_csv(output_path, index=False)
        print(f"Saved {sensor} features to '{output_path}'")
        merged_dfs[sensor] = merged

    return merged_dfs


# Main execution
if __name__ == "__main__":
    project_dir = r"C:\Users\User\Documents\project2_hishresifa"
    try:
        sensor_dfs, output_dir = load_sensor_data(project_dir)
        quest_df = load_questionnaire_data(project_dir)
        sensor_dfs = handle_missing_values(sensor_dfs, output_dir)
        feature_dfs = compute_features(sensor_dfs, quest_df)
        merged_dfs = merge_features(feature_dfs, quest_df, output_dir)
        print("Processing completed successfully.")
    except Exception as e:
        print(f"Error during processing: {e}")