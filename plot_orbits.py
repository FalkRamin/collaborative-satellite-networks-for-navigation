import numpy as np
import math
import matplotlib.pyplot as plt

def plot_earth_and_satellites(user_position, best_combination, satellite_position, visible_satellites):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Earth as a sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 6371 * np.outer(np.cos(u), np.sin(v))
    y = 6371 * np.outer(np.sin(u), np.sin(v))
    z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z, color='grey', alpha=0.3, rstride=4, cstride=4)

    # Plot latitude and longitude lines for better visualization
    latitudes = np.linspace(-90, 90, 19)  # Latitude lines from -90 to 90 degrees
    longitudes = np.linspace(-180, 180, 37)  # Longitude lines from -180 to 180 degrees

    for lat in latitudes:
        lat_rad = np.deg2rad(lat)
        x_lat = 6371 * np.cos(u) * np.cos(lat_rad)
        y_lat = 6371 * np.sin(u) * np.cos(lat_rad)
        z_lat = 6371 * np.sin(lat_rad)
        ax.plot(x_lat / 1000, y_lat / 1000, z_lat / 1000, color='black', linestyle='--', alpha=0.5)

    for lon in longitudes:
        lon_rad = np.deg2rad(lon)
        x_lon = 6371 * np.cos(lon_rad) * np.sin(v)
        y_lon = 6371 * np.sin(lon_rad) * np.sin(v)
        z_lon = 6371 * np.cos(v)
        ax.plot(x_lon / 1000, y_lon / 1000, z_lon / 1000, color='black', linestyle='--', alpha=0.5)

    # Plot the user position on the Earth's surface
    ax.scatter(user_position.user_x/1000, user_position.user_y/1000, user_position.user_z/1000, color='red', s=100, label='User Position')

    # Plot satellites from the best combination
    for sat in best_combination:
        sat_x_km, sat_y_km, sat_z_km = sat['x'] / 1000, sat['y'] / 1000, sat['z'] / 1000
        ax.scatter(sat_x_km, sat_y_km, sat_z_km, color='green', s=150, label=f'Sat {sat["satellite_index"]}')

        # Label each satellite
        ax.text(sat_x_km, sat_y_km, sat_z_km, f'Sat {sat["satellite_index"]}', color='black')

        user_x_km = user_position.user_x / 1000
        user_y_km = user_position.user_y / 1000
        user_z_km = user_position.user_z / 1000
        # Draw vector from user to satellite
        ax.quiver(
            user_x_km, user_y_km, user_z_km,
            sat_x_km - user_x_km,
            sat_y_km - user_y_km,
            sat_z_km - user_z_km,
            color='red', arrow_length_ratio=0.05, lw=1.5
        )

    # Plot all other visible satellites in black
    for sat in visible_satellites:
        if sat not in best_combination:  # Ensure it's not already plotted
            ax.scatter(sat['x'] / 1000, sat['y'] / 1000, sat['z'] / 1000, color='black', s=50)

    axis_length = 15000  # Length of the axis vectors (in km)
    # X-axis
    ax.quiver(0, 0, 0, axis_length, 0, 0, color='blue', arrow_length_ratio=0.05, lw=2)
    # Y-axis
    ax.quiver(0, 0, 0, 0, axis_length, 0, color='blue', arrow_length_ratio=0.05, lw=2)
    # Z-axis
    ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', arrow_length_ratio=0.05, lw=2)

    ax.set_box_aspect([1, 1, 1]) # Aspect ratio is 1:1:1

    # Set plot limits
    ax.set_xlim([-20000, 20000])
    ax.set_ylim([-20000, 20000])
    ax.set_zlim([-20000, 20000])

    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization')
    ax.legend(loc='best')
    plt.savefig("3D_Visualization_with_Axes.png")
    plt.show()

def set_axes_equal(ax):
    """Set equal scaling for all axes in a 3D plot."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    max_range = 0.5 * np.ptp(limits, axis=1).max()
    midpoints = np.mean(limits, axis=1)
    ax.set_xlim3d([midpoints[0] - max_range, midpoints[0] + max_range])
    ax.set_ylim3d([midpoints[1] - max_range, midpoints[1] + max_range])
    ax.set_zlim3d([midpoints[2] - max_range, midpoints[2] + max_range])

# Function to generate 3D plot of the user points
def plot_3d_distribution(gdop_map, num_lat, num_lon):
    # Create a grid of latitude and longitude points
    latitudes = np.linspace(-90, 90, num_lat)
    longitudes = np.linspace(-180, 180, num_lon, endpoint=False) # Avoid including the endpoint

    # Convert lat/lon to Cartesian coordinates (assuming Earth radius = 6371 km)
    earth_radius = 6371000
    X, Y, Z = [], [], []
    colors = []

    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            x = earth_radius * np.cos(np.radians(lat)) * np.cos(np.radians(lon))
            y = earth_radius * np.cos(np.radians(lat)) * np.sin(np.radians(lon))
            z = earth_radius * np.sin(np.radians(lat))

            X.append(x)
            Y.append(y)
            Z.append(z)
            colors.append(gdop_map[i, j])

    # Plotting the 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X, Y, Z, color='blue', marker='o', s=10)

    # Set labels and title
    ax.set_title('3D Distribution of User Points and GDOP Values')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    # Set equal scaling for axes
    set_axes_equal(ax)

    plt.savefig(f"3D_gdop_plot.png")
    plt.show()

