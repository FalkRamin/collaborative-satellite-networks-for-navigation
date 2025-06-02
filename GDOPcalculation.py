from typing import List, Any

import numpy as np
import math
from itertools import combinations
import matplotlib.pyplot as plt

class UserPosition:
    def __init__(self, user_x, user_y, user_z):
        self.user_x = user_x
        self.user_y = user_y
        self.user_z = user_z

def calculate_angle(user_position, satellite_position):

    user_vector = np.array([user_position.user_x, user_position.user_y, user_position.user_z])
    satellite_vector = np.array([satellite_position['x'], satellite_position['y'], satellite_position['z']])

    # Vector from user's position to satellite
    user_to_satellite_vector = satellite_vector - user_vector

    dot_product = np.dot(user_vector, user_to_satellite_vector)
    user_vector_magnitude = np.linalg.norm(user_vector)
    user_to_satellite_magnitude = np.linalg.norm(user_to_satellite_vector)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (user_vector_magnitude * user_to_satellite_magnitude)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure value is within valid range for arccos
    angle_radians = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees

def calculate_gdop(user_position, satellite_positions):
    if len(satellite_positions) < 4:
        raise ValueError("At least 4 satellites are required to calculate GDOP")

    min_gdop = float('inf')
    best_combination = None
    visible_satellites: list[Any] = []

    #Filter for only visible sats
    for satellite in satellite_positions:
        angle = calculate_angle(user_position, satellite)
        if angle <= 75:  # 90 Ref:"75deg COMPARISON OF AVAILABILITY OF GALILEO, GPS AND A COMBINED GALILEO/GPS NAVIGATION SYSTEMS"Satellite is considered visible if the angle is 90 degrees or less
            visible_satellites.append(satellite)

    # print("Number of visible satellites:", len(visible_satellites))

    # Iterate through all combinations of 4 satellites
    for combination in combinations(visible_satellites, 4):
        P = np.zeros((4, 4))

        for i, sat in enumerate(combination):
            x, y, z = sat['x'], sat['y'], sat['z']
            r = math.sqrt(
                   (x - user_position.user_x) ** 2 + (y - user_position.user_y) ** 2 + (z - user_position.user_z) ** 2)
            P[i, 0] = (x - user_position.user_x) / r
            P[i, 1] = (y - user_position.user_y) / r
            P[i, 2] = (z - user_position.user_z) / r
            P[i, 3] = 1

        try:
            # Calculate the transpose of P
            P_T = np.transpose(P)
            # Calculate the inverse of P
            #P_inv = np.linalg.inv(P)
            #P_inv_pseudo = np.linalg.pinv(P) #Pseudo inverse for numerical stability (GTP)

            D = np.dot(P_T, P)
            T = np.linalg.inv(D)
            # GDOP is the square root of the sum of the diagonal elements of D
            gdop = np.sqrt(np.trace(T))

            if gdop < min_gdop:
                min_gdop = gdop
                best_combination = combination

        except np.linalg.LinAlgError:
            # Skip this combination if the matrix is singular
            continue

    if best_combination is None:
        raise ValueError("Could not find a valid combination of satellites to calculate GDOP")

    return min_gdop, best_combination, visible_satellites

# New function to calculate GDOP for multiple points
def calculate_gdop_for_grid(satellite_positions, num_lat, num_lon): #num_lat=180 num_lFon=36
    gdop_map = np.zeros((num_lat, num_lon))
    satellite_count_map = np.zeros((num_lat, num_lon)) # To store the number and positions of the satellites

    gdop_values = []

    for i, lat in enumerate(np.linspace(-90, 90, num_lat)):
        for j, lon in enumerate(np.linspace(-180, 180, num_lon)):
            # Convert lat, lon to Cartesian coordinates (assuming Earth radius = 6371 km) [changed into meters]
            x = 6371000 * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
            y = 6371000 * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
            z = 6371000 * np.sin(np.radians(lat))

            user_position = UserPosition(x, y, z)
            gdop, _, visible_satellites = calculate_gdop(user_position, satellite_positions)
            if gdop is not None:
                gdop_map[i, j] = gdop
                satellite_count_map[i,j] = len(visible_satellites)
                gdop_values.append(gdop)
            else:
                gdop_map[i, j] = np.nan  # Assign NaN if GDOP cannot be calculated
                satellite_count_map[i, j] = 0
    gdop_values = np.array(gdop_values)

    return gdop_map, satellite_count_map, num_lat, num_lon

# Function to plot the GDOP map
def plot_gdop_map(gdop_map):
    plt.figure(figsize=(12, 6))
    plt.imshow(gdop_map, extent=[-180, 180, -90, 90], origin='lower', cmap='jet', vmin=1, vmax=4)
    cbar = plt.colorbar(label='GDOP Value')
    cbar.set_ticks(np.linspace(1, 4, num=10))
    plt.title('GDOP Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f"GDOP_Map.png")
    plt.show()

def plot_satellites_over_latitude(satellite_count_map):
    # Average the number of satellites over longitude for each latitude
    avg_satellites_per_lat = np.nanmean(satellite_count_map, axis=1)
    latitudes = np.linspace(-90, 90, satellite_count_map.shape[0]) #
    plt.figure(figsize=(10, 6))
    plt.plot(latitudes, avg_satellites_per_lat, color='blue', marker='o', linestyle='-', linewidth=2, markersize=5)
    plt.title('Average Number of Visible Satellites Over Latitude')
    plt.xlabel('Latitude (degrees)')
    plt.ylabel('Number of Visible Satellites')
    plt.grid(True)
    plt.savefig(f"SatelliteNumber_Over_Latitude.png")
    plt.show()
def calculate_elevation(user_position, sat_position):
    # Vector from user to satellite
    dx = sat_position['x'] - user_position.user_x
    dy = sat_position['y'] - user_position.user_y
    dz = sat_position['z'] - user_position.user_z

    # Distance from the user to satellite
    r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Unit vector in the z-direction (local vertical)
    vertical_unit_vector = np.array([0, 0, 1])

    # Unit vector from user to satellite
    satellite_unit_vector = np.array([dx / r, dy / r, dz / r])

    # Elevation angle is the angle between the satellite unit vector and the vertical direction
    dot_product = np.dot(satellite_unit_vector, vertical_unit_vector)

    # Calculate elevation (arcsin gives the angle in radians)
    elevation = np.degrees(np.arcsin(dot_product))
    return elevation
def calculate_azimuth(user_position, sat_position):
    # Vector from user to satellite
    dx = sat_position['x'] - user_position.user_x
    dy = sat_position['y'] - user_position.user_y
    dz = sat_position['z'] - user_position.user_z

    # Azimuth is calculated in the XY-plane
    azimuth = np.degrees(np.arctan2(dy, dx))  # Use arctan2 for quadrant correction
    return azimuth
def calculate_spherical_coordinates_receiver(satellite_pos, user_position):
    """Calculate spherical coordinates relative to the receiver's local system."""
    # Use the attribute access for the receiver position
    x_receiver = user_position.user_x
    y_receiver = user_position.user_y
    z_receiver = user_position.user_z

    # Calculate relative positions
    x_rel = satellite_pos['x'] - x_receiver
    y_rel = satellite_pos['y'] - y_receiver
    z_rel = satellite_pos['z'] - z_receiver

    # Calculate r, theta, and phi in the receiver's local system
    r = math.sqrt(x_rel ** 2 + y_rel ** 2 + z_rel ** 2)
    theta = math.atan2(y_rel, x_rel)  # Azimuth angle
    theta_degrees = math.degrees(theta) # Convert to degrees
    if theta_degrees < 0:
        theta_degrees += 360 # Normalize to [0, 360]

    phi = math.asin(z_rel / r)  # Elevation angle relative to receiver's horizon
    phi_degrees = math.degrees(phi)

    # Convert angles to degrees and return
    return r, theta_degrees, phi_degrees

def calculate_spherical_coordinates_earth_origin(x, y, z):
    """Calculate spherical coordinates relative to Earth's origin."""
    r = math.sqrt(x**2 + y**2 + z**2)
    theta = math.atan2(y, x)  # Azimuth angle in radians
    theta_degrees = math.degrees(theta)
    if theta_degrees < 0:
        theta_degrees += 360

    phi = math.acos(z / r)    # Elevation angle in radians from z-axis (polar angle)
    phi_degrees = math.degrees(phi)
    return r, theta_degrees, phi_degrees  # Return angles in degrees
