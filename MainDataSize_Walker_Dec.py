import time
import matplotlib.pyplot as plt
from PropagationWithAttitude import Satellite
from GDOPcalculation import calculate_gdop, UserPosition
from plot_orbits import plot_earth_and_satellites
from GDOPcalculation import calculate_gdop_for_grid, plot_gdop_map, calculate_azimuth, calculate_elevation, calculate_spherical_coordinates_receiver, calculate_spherical_coordinates_earth_origin
#---------------------------------------------------------------------------------------------------------------------#
# Set the following parameter as input
#---------------------------------------------------------------------------------------------------------------------#

#-------------------------#
# Configuration parameters
#-------------------------#
# Satellite Constellation parameters
number_satellites = [30]
number_planes = 3
orbit_template = {
    'semimajoraxis': 26571.0,                   # kilometer
    'inclination': 56.0,                        # degrees
    'eccentricity': 0.001,
    'arg_of_perigee': 0.0,
}
# Simulation parameters
simulation_duration = 86400                     # seconds
time_step = 1                                   # seconds

# User/Receiver parameter
elevation_mask_deg = 5.0                        # degree
user_position = UserPosition(6371000, 0, 0)     # meter

#-------------------------#
# Output parameters
#-------------------------#
# GDOP plot vs time
plot_gdop_vs_time_flag = True                   # set True or False

# GDOP map - only created at the last time step of the simulation duration
gdop_map_flag = False                           # set True or False
number_latitude = 100                           # must be an integer
number_longitude = 200                          # must be an integer

# Creats a 3D plot visualizing the Earth, Sats, Receiver - only for the last time step of the simulation
earth_sat_3d_flag = False                       # set True or False

#---------------------------------------------------------------------------------------------------------------------#

def update_plot(frame_num, satellites, scatter):
    # Propagate all the satellites
    for satellite in satellites:
        satellite.propagate(time_step)

    # Extract x, y, z coordinates of the satellites
    x_coords = [satellite.orbit['x'] for satellite in satellites]
    y_coords = [satellite.orbit['y'] for satellite in satellites]
    z_coords = [satellite.orbit['z'] for satellite in satellites]

    # Update the position of the scatter plot
    scatter._offsets3d = (x_coords, y_coords, z_coords)

if __name__ == "__main__":
    Num_Sat=number_satellites
    AvgPerN=[]
    all_gdop_values = []

    # Open a new file to log visible satellites
    visible_satellites_file = open('visible_satellites_output.txt', 'w')

    # Create the output file to store GDOP information
    with open('gdop_output.txt', 'w') as output_file:
        output_file.write("Time Step, GDOP, Best Satellites (Index, X, Y, Z, Azimuth, Elevation, r_Earth, theta_Earth, phi_Earth, r_Receiver, theta_Receiver, phi_Receiver)\n")

        for numsat in Num_Sat:
            total_times=[]

        for iterAver in range(1):
        
            num_satellites = numsat
            num_planes=number_planes
            num_satellites_per_plane=num_satellites/num_planes
            
            print("Num sat is", num_satellites)
            satellites = [Satellite(num_satellites_per_plane, num_planes, i, orbit=orbit_template) for i in range(num_satellites)]

            countersat=0

            # propagate the satellites for a given time duration, with a specific time step
            #time_step = 1  # seconds
            duration = simulation_duration
            num_steps = int(duration / time_step)
        
            start_time = time.time()
            total_time = None

            breaker = False
            gdop_val = []

            #Here we iterate on the simulation time
            for k in range(num_steps):
                if breaker:
                    break
                # propagate all the satellites
                for satellite in satellites:
                    satellite.propagate(time_step)

                satellite_positions = []

                # Here we enter in a loop of a specific time step, reiterated for all the couples of satellites 
                for i in range(len(satellites)):
                    #It starts from i+1 so that a couple of sats i,j is not considered twice

                    # Extract x, y, z coordinates of the satellite
                    x_coords = satellites[i].orbit['x']
                    y_coords = satellites[i].orbit['y']
                    z_coords = satellites[i].orbit['z']
                    satellite_positions.append({'satellite_index': i, 'x': x_coords, 'y': y_coords, 'z': z_coords})

                # Calculate and print GDOP
                if len(satellite_positions) >= 4:
                    try:
                        min_gdop, best_combination, visible_satellites = calculate_gdop(user_position, satellite_positions, elevation_mask_deg)
                        gdop_val.append({'time_step': k, 'min_gdop':min_gdop})
                        all_gdop_values.append(min_gdop)  # <-- Collecting all GDOP values

                        # Write GDOP and satellite info to the output file
                        output_file.write(f"--------------------------------------------------\n")
                        output_file.write(f"Time Step: {k}, GDOP: {min_gdop}\n")
                        output_file.write("Best Satellite Combination:\n")

                        # Write the visible satellites to the new file
                        visible_satellites_file.write(
                            f"Time Step: {k}, Visible Satellites: {len(visible_satellites)}\n")
                        for sat in visible_satellites:
                            azimuth = calculate_azimuth(user_position, sat)
                            elevation = calculate_elevation(user_position, sat)
                            x, y, z = sat['x'], sat['y'], sat['z']

                            # Earth-relative spherical coordinates
                            r_earth, theta_earth, phi_earth = calculate_spherical_coordinates_earth_origin(x, y, z)

                            # Receiver-relative spherical coordinates
                            r_receiver, theta_receiver, phi_receiver = calculate_spherical_coordinates_receiver(sat,
                                                                                                                user_position)

                            visible_satellites_file.write(
                                f"Satellite {sat['satellite_index']} - x: {x}, y: {y}, z: {z}, "
                                f"Azimuth: {azimuth:.2f}, Elevation: {elevation:.2f}, "
                                f"r_Earth: {r_earth:.2f}, theta_Earth: {theta_earth:.2f}, phi_Earth: {phi_earth:.2f}, "
                                f"r_Receiver: {r_receiver:.2f}, theta_Receiver: {theta_receiver:.2f}, phi_Receiver: {phi_receiver:.2f}\n"
                            )

                        for sat in satellite_positions:
                            azimuth = calculate_azimuth(user_position, sat)
                            elevation = calculate_elevation(user_position, sat)

                        for sat in best_combination:
                            azimuth = calculate_azimuth(user_position, sat)
                            elevation = calculate_elevation(user_position, sat)
                            # Extract x, y, z coordinates from the satellite dictionary
                            x, y, z = sat['x'], sat['y'], sat['z']
                            # Earth-relative spherical coordinates
                            r_earth, theta_earth, phi_earth = calculate_spherical_coordinates_earth_origin(sat['x'], sat['y'], sat['z'])

                            # Receiver-relative spherical coordinates
                            r_receiver, theta_receiver, phi_receiver = calculate_spherical_coordinates_receiver(sat, user_position)

                            output_file.write(
                                f"Satellite {sat['satellite_index']} - x: {sat['x']}, y: {sat['y']}, z: {sat['z']}, "
                                f"Azimuth: {azimuth:.2f}, Elevation: {elevation:.2f}, "
                                f"r_Earth: {r_earth:.2f}, theta_Earth: {theta_earth:.2f}, phi_Earth: {phi_earth:.2f}, "
                                f"r_Receiver: {r_receiver:.2f}, theta_Receiver: {theta_receiver:.2f}, phi_Receiver: {phi_receiver:.2f}\n"
                            )

                    except ValueError as e:
                        print(f"\nError calculating GDOP: {e}")
                        output_file.write(f"\nError calculating GDOP at step {k}: {e}\n")
                else:
                    print("\nNot enough satellite data to calculate GDOP. At least 4 satellites are required.")
                    output_file.write(f"\nNot enough satellite data to calculate GDOP at step {k}\n")

        # Calculate overall GDOP statistics
        if all_gdop_values:
            avg_gdop = sum(all_gdop_values) / len(all_gdop_values)
            max_gdop = max(all_gdop_values)
            min_gdop = min(all_gdop_values)

            # Write the GDOP statistics to the output file
            output_file.write("\nGDOP Statistics:\n")
            output_file.write(f"Average GDOP: {avg_gdop:.2f}\n")
            output_file.write(f"Maximum GDOP: {max_gdop:.2f}\n")
            output_file.write(f"Minimum GDOP: {min_gdop:.2f}\n")


    time_steps = [entry['time_step'] for entry in gdop_val]
    gdop_values = [entry['min_gdop'] for entry in gdop_val]

    # Create a GDOP plot over time
    if(plot_gdop_vs_time_flag):
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, gdop_values, marker='o', linestyle='-', color='b')

        # Add titles and labels
        plt.title('GDOP')
        plt.xlabel('Time Step')
        plt.ylabel('Minimum GDOP')
        plt.grid(True)

        # Show the plot
        plt.savefig(f"gdop_plot.png")
        plt.show()
        plt.close()

    # Plot Earth, user, and satellites
    if(earth_sat_3d_flag):
        plot_earth_and_satellites(user_position, best_combination, satellite_positions, visible_satellites)

    # GDOP map and plot
    if(gdop_map_flag):
        gdop_map, satellite_count_map, num_lat, num_lon = calculate_gdop_for_grid(satellite_positions, number_latitude, number_longitude, elevation_mask_deg)
        plot_gdop_map(gdop_map)

    print(f"Maximum GDOP: {max_gdop}")
    print(f"Minimum GDOP: {min_gdop}")
    print(f"Average GDOP: {avg_gdop}")
