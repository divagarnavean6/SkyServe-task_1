import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from datetime import datetime

data = pd.read_csv("sample_data.csv")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

threshold_distance = 10

data['lat_rad'] = np.radians(data['lat'])
data['lon_rad'] = np.radians(data['lon'])

tree = cKDTree(data[['lat_rad', 'lon_rad']])

def find_nearby_vessels(lat, lon, threshold_distance):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    nearby_indices = tree.query_ball_point([lat_rad, lon_rad], threshold_distance / 6371.0)
    return data.iloc[nearby_indices]['mmsi'].tolist()
proximity_events = []
for index, row in data.iterrows():
    nearby_vessels = find_nearby_vessels(row['lat'], row['lon'], threshold_distance)
    nearby_vessels.remove(row['mmsi'])  # Remove self from nearby vessels
    if nearby_vessels:
        proximity_events.append({
            'mmsi': row['mmsi'],
            'vessel_proximity': nearby_vessels,
            'timestamp': row['timestamp']
        })

proximity_df = pd.DataFrame(proximity_events)

print(proximity_df)
