**Vessel Proximity Detection**

**Overview**

This project is designed to detect vessel proximity events, where two vessels with different MMSIs (Maritime Mobile Service Identity) come within a specified threshold distance during a given time frame. The algorithm uses the Haversine formula for distance calculations and leverages efficient spatial querying techniques with a KD-tree for better performance.

**Features**

Calculates distances using the Haversine formula.
Efficient spatial querying with KD-tree.
Detects and outputs vessel proximity events within a specified threshold distance and time frame

**Parameters**

threshold_distance: The distance threshold (in kilometers) to consider for proximity events.
time_frame_start: The start of the time frame (inclusive) in the format 'YYYY-MM-DD HH:MM:SS'.
time_frame_end: The end of the time frame (inclusive) in the format 'YYYY-MM-DD HH:MM:SS'.

**Output**

The script will output a DataFrame with the following columns:

mmsi: The MMSI of the vessel.
vessel_proximity: A list of MMSIs of vessels with which it interacts within the threshold distance.
timestamp: The timestamp of the proximity event
