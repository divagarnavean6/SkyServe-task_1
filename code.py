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
    # Filter data based on time frame
    filtered_data = dataset[(dataset['Timestamp'] >= time_frame_start) & (dataset['Timestamp'] <= time_frame_end)].copy()
    
    # Convert latitude and longitude to radians
    filtered_data['Latitude_rad'] = np.radians(filtered_data['Latitude'])
    filtered_data['Longitude_rad'] = np.radians(filtered_data['Longitude'])
    
    # Create a quadtree (using KD-tree for simplicity)
    coords = filtered_data[['Latitude_rad', 'Longitude_rad']].values
    tree = cKDTree(coords)
    
    # Convert threshold distance to radians
    threshold_distance_rad = threshold_distance / 6371.0
    
    proximity_dict = {mmsi: [] for mmsi in filtered_data['MMSI']}
    
    for index, row in filtered_data.iterrows():
        # Query the KD-tree to find nearby vessels within the threshold distance
        nearby_indices = tree.query_ball_point([row['Latitude_rad'], row['Longitude_rad']], threshold_distance_rad)
        
        for idx in nearby_indices:
            if idx != index:
                mmsi1 = row['MMSI']
                mmsi2 = filtered_data.iloc[idx]['MMSI']
                if mmsi1 != mmsi2:
                    actual_distance = haversine_distance(row['Latitude'], row['Longitude'],
                                                         filtered_data.iloc[idx]['Latitude'], filtered_data.iloc[idx]['Longitude'])
                    if actual_distance < threshold_distance:
                        proximity_dict[mmsi1].append(mmsi2)
    
    proximity_events = []
    for mmsi, proximities in proximity_dict.items():
        if proximities:
            timestamp = filtered_data[filtered_data['MMSI'] == mmsi]['Timestamp'].iloc[0]
            proximity_events.append({'mmsi': mmsi, 'vessel_proximity': proximities, 'timestamp': timestamp})
    
    return pd.DataFrame(proximity_events)

# Example usage:
# Load dataset from CSV file
dataset = pd.read_csv('sample_data.csv')
dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

# Specify threshold distance (in kilometers) and time frame
threshold_distance = 6000  # Set as needed
time_frame_start = '2024-01-01 00:00:00'
time_frame_end = '2024-01-02 00:00:00'

# Find proximity events
proximity_events_df = find_proximity_events(dataset, threshold_distance, time_frame_start, time_frame_end)

# Print proximity events
print(proximity_events_df)
