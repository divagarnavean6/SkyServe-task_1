import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from scipy.spatial import cKDTree

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def find_proximity_events(dataset, threshold_distance, time_frame_start, time_frame_end):
    
    filtered_data = dataset[(dataset['timestamp'] >= time_frame_start) & (dataset['timestamp'] <= time_frame_end)].copy()
    
    filtered_data['lat_rad'] = np.radians(filtered_data['lat'])
    filtered_data['log_rad'] = np.radians(filtered_data['log'])
    
    coords = filtered_data[['lat_rad', 'log_rad']].values
    tree = cKDTree(coords)
    
    threshold_distance_rad = threshold_distance / 6371.0
    
    proximity_dict = {mmsi: [] for mmsi in filtered_data['mmsi']}
    
    for index, row in filtered_data.iterrows():
        
        nearby_indices = tree.query_ball_point([row['lat_rad'], row['log_rad']], threshold_distance_rad)
        
        for idx in nearby_indices:
            if idx != index:
                mmsi1 = row['mmsi']
                mmsi2 = filtered_data.iloc[idx]['mmsi']
                if mmsi1 != mmsi2:
                    actual_distance = haversine_distance(row['lat'], row['log'],
                                                         filtered_data.iloc[idx]['lat'], filtered_data.iloc[idx]['log'])
                    if actual_distance < threshold_distance:
                        proximity_dict[mmsi1].append(mmsi2)
    
    proximity_events = []
    for mmsi, proximities in proximity_dict.items():
        if proximities:
            timestamp = filtered_data[filtered_data['mmsi'] == mmsi]['timestamp'].iloc[0]
            proximity_events.append({'mmsi': mmsi, 'vessel_proximity': proximities, 'timestamp': timestamp})
    
    return pd.DataFrame(proximity_events)


dataset = pd.read_csv('sample_data.csv')
dataset.columns = ['mmsi', 'timestamp', 'lat', 'log']
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], utc=True)


threshold_distance = 5  # Set as needed
time_frame_start = pd.to_datetime('2024-01-01 00:00:00', utc=True)
time_frame_end = pd.to_datetime('2024-01-02 00:00:00', utc=True)


proximity_events_df = find_proximity_events(dataset, threshold_distance, time_frame_start, time_frame_end)


print(proximity_events_df)
