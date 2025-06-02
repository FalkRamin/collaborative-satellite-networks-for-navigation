import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt


from PropagationWithAttitude import Satellite, distance_between  # F:PropagationWalkerV2 changed
from CommSubsystemNEW import CommSubsystem
from plottingDiffDataDim import plot_avg_time_vs_num_satellites
from GDOPcalculation import calculate_gdop, UserPosition
from plot_orbits import plot_earth_and_satellites, plot_3d_distribution
from GDOPcalculation import calculate_gdop_for_grid, plot_gdop_map, plot_satellites_over_latitude, calculate_azimuth, calculate_elevation, calculate_spherical_coordinates_receiver, calculate_spherical_coordinates_earth_origin

# Satellite Constellation parameters 
number_satellites =[30]
number_planes = 3
simulation_duration = 15 #28500 #3600 (in sec)
number_latitude = 100
number_longitude = 200

# def update_adjacency_matrices(adj_matrices, adj_matrix):
#     # Save the current adjacency matrix in the list
#     adj_matrices.append(np.copy(adj_matrix))

#     # Keep only the last 10 adjacency matrices
#     if len(adj_matrices) > 10:
#         adj_matrices.pop(0)

# The rest of your code related to generating and analyzing data
# ...


# This code evaluate if there has been an exchange of information among satellites, considering:
# - Processing time of the receiver 
# - Initially, only one satellite has the information 
# - One satellites can communicate with everyone, but depending on the setting of the commm subsystem, all of them can communicate with each other, in this case the relay is useless
# - The sats communicate if the effective data rate is more than 0 and if one of the 2 has the information and they have the same band or one of them is the relay node
# - It is supposed to work for decentralized or decentralized or distributed approach, depending on the setting of the comm subsystem 
# - In this version, it is possible to set the minimum amount of data to be sent. (You can set the data in the variable Data_Size)
# - The value of the Adjacency matrix [i][j] is set to 1 only if the complete data is sent
# - The data_matrix consider that the data package has to be sent with continuous communication
# - The data_matrix_acc consider the total data potentially exchanged between couple of satellites for each time they meet each other
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
    # Define the number of satellites to be propagated
    
    Num_Sat=number_satellites # F:60,70,80,90,100,150,200,250,300,500,1000,10000]
 
    AvgPerN=[]
    all_gdop_values = []
    # Open a new file to log visible satellites
    visible_satellites_file = open('visible_satellites_output.txt', 'w')

    # Create the output file to store GDOP information
    with open('gdop_output.txt', 'w') as output_file:
        output_file.write("Time Step, GDOP, Best Satellites (Index, X, Y, Z, Azimuth, Elevation, r_Earth, theta_Earth, phi_Earth, r_Receiver, theta_Receiver, phi_Receiver)\n")

        for numsat in Num_Sat:
            total_times=[]
        
        # If more iteration for the same number of satellites want to be performed, change the the "range" for the iterAver
        # This will also evaluate the average with all the iterations 

        for iterAver in range(1):
        
            num_satellites = numsat
            num_planes=number_planes # Galileo: 3
            num_satellites_per_plane=num_satellites/num_planes
            
            print("Num sat is", num_satellites)
            # Recall the class Satellite to propagate the N satellites
            satellites = [Satellite(num_satellites_per_plane, num_planes, i=i) for i in range(num_satellites)]
            #satellites = [Satellite(num_satellites=numsat, altitude=800, inclination=45, spacing=15, i=i) for i in range(num_satellites)]
            #satellites = [Satellite() for i in range(num_satellites)]
            #Extracting x, y, and z coordinates of the satellites

            countersat=0

            adjacency_matrices = []  # List to store the last 10 adjacency matrices
            # Initialize the adjacency matrix
            # adjacency_matrix = np.zeros((num_satellites, num_satellites), dtype=int)

            # set the flag of one random satellite to True
            # This flag represents the knowledge of the information (1 if the information is obtained)
            # Usually just one satellite knows the info
            satellites[random.randint(0, num_satellites - 1)].flag = 1
            
            # set one random satellite to act as relay - it correspond to assign to the band the value of 5
            # All the other band are defined in the Class Satellite in the propagation.py
            satellites[random.randint(0, num_satellites - 1)].commsys['band'] = 5

            # propagate the satellites for a given time duration, with a specific time step
            time_step = 1  # seconds
            duration = simulation_duration # 86400 # 360 # 86400  # seconds (in one day 86400)
            num_steps = int(duration / time_step)
        
            start_time = time.time()
            total_time = None

            #Link distance is useful only if the communication does not depend on the effective data rate and sensitivity of the receiver but just from a euclidian distance
            adj_matrix_acc = np.zeros((num_satellites, num_satellites), dtype=int)
            breaker = False #our mighty loop exiter!
            
            #Initialize data Matrix, the data is set to zero if there is not continuous communication
            data_matrix= np.zeros((num_satellites, num_satellites), dtype=int)

            #Initialize data Matrix, the data is accumulated to check the total quantity of data exchange between 2 sats
            data_matrix_acc= np.zeros((num_satellites, num_satellites), dtype=int)

            # Initialize the set to keep track of communicated satellites in the current time step
            #communicated_satellites = set()
            
            # Introduce a processing time for a satellite after it receives information
            processing_time = 10 # You can adjust this value based on your requirements (in seconds)

            #Introduce what is the size of the data I want to send
            Data_Size= 1680 #[bits]

            ### Falk ###
            gdop_val = []
            ### Falk ###

            #Here we iterate on the simulation time
            for k in range(num_steps):
                if breaker: # the interesting part
                    break

                # Reset the set of communicated satellites at the beginning of each time step (It is not suggested to use it, but consider the processing time)
                #communicated_satellites.clear()


                # propagate all the satellites
                for satellite in satellites:
                    satellite.propagate(time_step)
                    
                    if satellite.processing_time > 0:
                            satellite.processing_time -= time_step
                    if satellite.processing_time < 0:
                            satellite.processing_time =0                      
                # Assuming 'satellites' is a list containing Satellite objects with position information
                # Set up the plot
                #     # Set up the plot
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')

                # # Set fixed axes limits
                # ax.set_xlim([-10000000, 10000000])
                # ax.set_ylim([-10000000, 10000000])
                # ax.set_zlim([-10000000, 10000000])

                # # Initialize scatter plot
                # scatter = ax.scatter([], [], [], c='b', marker='o')

                # # Set up the animation
                # anim = animation.FuncAnimation(fig, update_plot, fargs=(satellites, scatter), frames=100, interval=100)

                # # Show the animation
                # plt.show()
                # # Extract x, y, z coordinates of the satellites
                # x_coords = [satellite.orbit['x'] for satellite in satellites]
                # y_coords = [satellite.orbit['y'] for satellite in satellites]
                # z_coords = [satellite.orbit['z'] for satellite in satellites]

                # # Plot the satellites in 3D
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax.scatter(x_coords, y_coords, z_coords, c='b', marker='o')

                # # Set labels and title
                # ax.set_xlabel('X (m)')
                # ax.set_ylabel('Y (m)')
                # ax.set_zlabel('Z (m)')
                # ax.set_title('Satellite Positions')

                # plt.show()
                adj_matrix = np.zeros((num_satellites, num_satellites), dtype=int)

                # ## Falk ## #
                satellite_positions = []
                user_position = UserPosition(6371000, 0, 0) #User position in meters
                # ## Falk ## #

                # Here we enter in a loop of a specific time step, reiterated for all the couples of satellites 
                for i in range(len(satellites)):
                    #It starts from i+1 so that a couple of sats i,j is not considered twice

                    ### Falk ###
                    # Extract x, y, z coordinates of the satellite
                    x_coords = satellites[i].orbit['x']
                    y_coords = satellites[i].orbit['y']
                    z_coords = satellites[i].orbit['z']
                    satellite_positions.append({'satellite_index': i, 'x': x_coords, 'y': y_coords, 'z': z_coords})
                    ### Falk ###

                    for j in range(i + 1, len(satellites)):

                        dist = distance_between(satellites[i], satellites[j], time_step)
                        data_comms = CommSubsystem()
                        eff_datarate = data_comms.calculateEffectiveDataRate(dist)

                        # if the data rate is more than 0, the satellites can communicate.
                        # and if one of the 2 satellite has the information (One has the flag egual to 0 and the other to 1)
                        if eff_datarate > 0 and satellites[i].flag != satellites[j].flag and satellites[i].processing_time==satellites[j].processing_time==0:
                            #if both satellites has the same band or one of them is the "relay node"
                            if satellites[i].commsys['band']==satellites[j].commsys['band'] or satellites[i].commsys['band']==5 or satellites[j].commsys['band']==5:
                                #If all these conditions are respected, the two satellites exchanged info
                                # Set up the value of the flag to 1
                                if satellites[i].flag==1:

                                    data_matrix[i][j]=data_matrix[j][i]=data_matrix[i][j]+(eff_datarate*time_step)
                                    data_matrix_acc[i][j]=data_matrix_acc[j][i]=data_matrix_acc[i][j]+(eff_datarate*time_step)
                                    if data_matrix[i][j] > Data_Size:
                                        satellites[j].flag = 1
                                        adj_matrix_acc[i][j] = 1
                                        adj_matrix_acc[j][i] = 1
                                        adj_matrix[i][j] = 1
                                        adj_matrix[j][i] = 1


                                        # When the satellite receives the message need processing time before it can receive more or can send it to someone else

                                        satellites[j].processing_time = processing_time  # Set processing time for the satellite

                                #print(data_matrix)                                    

                                elif satellites[j].flag == 1:
                                    
                                    data_matrix[i][j]=data_matrix[j][i]=data_matrix[i][j]+(eff_datarate*time_step)
                                    data_matrix_acc[i][j]=data_matrix_acc[j][i]=data_matrix_acc[i][j]+(eff_datarate*time_step)
                                    if data_matrix[i][j] > Data_Size:
                                        satellites[i].flag = 1
                                        adj_matrix_acc[i][j] = 1
                                        adj_matrix_acc[j][i] = 1
                                        adj_matrix[i][j] = 1
                                        adj_matrix[j][i] = 1
                                        # When the satellite receives the message need processing time before it can receive more or can send it to someone else

                                        satellites[i].processing_time = processing_time  # Set processing time for the satellite
                                #if(np.sum(adj_matrix))>0:
                                #   print(adj_matrix)

                            else:
                                #If 2 sats do not communicate in a specific time step, the data is set to zero
                                data_matrix[i][j]=data_matrix[j][i]=0
                        else: 
                            data_matrix[i][j]=data_matrix[j][i]=0
                #print("\nStored satellite positions:")  # F:Print Stored Sat. position:

                ### Falk ###
                # Calculate and print GDOP
                if len(satellite_positions) >= 4:
                    try:
                        min_gdop, best_combination, visible_satellites = calculate_gdop(user_position, satellite_positions)
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

                ### Falk ###

                #print(adj_matrix)
                # check if all satellites have the flag set to True, they all communicated, else they did not
                if all(satellite.flag for satellite in satellites):
                    #total_time = (i+1) * time_step
                    total_time=k*time_step
                    if total_time == None:
                        print("The information has not been spread along all the nodes")
                    else:
                        total_times.append(total_time)
                        print(f"The cumulative adjacency matrix for {len(satellites)} is:\n {adj_matrix_acc}")
                        adjacency_matrices.append(np.copy(adj_matrix))
                        filename = f'adjacency_matrix_{num_satellites}.npy'
                        np.save(filename, adj_matrix_acc)
                        #print(adjacency_matrices)     

                    # total_time=time.time()-start_time
                    # breaker = True
                    # break # adjusted to not end early

#Save a copy of the current adjacency matrix for the last 10 time steps
                adjacency_matrices.append(np.copy(adj_matrix))
                #if(np.sum(adj_matrix))>0:
                    #print(adj_matrix)
                #print(len(adjacency_matrices))
                if len(adjacency_matrices) > 10:
                   #print(len(adjacency_matrices))

                   adjacency_matrices.pop(0)

                # print("Last 10 adjacency matrices:")
                # for matrix in adjacency_matrices:
                #     print(matrix)



        # =============================================================================
        #To plot the sats that received the information
        #     for satellite in satellites:
        #         countersat=countersat+1
        #         print(satellite.flag)
        #         if satellite.flag==1:
        #             print("The satellite number ",countersat," has received the information")
        # =============================================================================
                        
            print(f"Total time to propagate the information: {total_time} seconds")
        

        
        end_time = time.time()
        #print(f"Total elapsed time: {end_time - start_time} seconds")
        
        avg_total_time=None
        # Find the average of the total times if it was desired and specified in the AvgPerN 
        if len(total_times)==0:
            print("The info has never been spread among all the satellites")
        else:
            print("The average has been evaluated in", len(total_times), "iterations")
            avg_total_time = sum(total_times) / len(total_times)
        print("Average total time:", avg_total_time)


        # Find the average of the total times
        if avg_total_time==0:
            print("The info has never been spread among all the satellites")
        else:
            print("The average has been evaluated for", Num_Sat, "satellites")
            AvgPerN.append(avg_total_time)
            
        print("Average total time per number of satellites:", AvgPerN)
        avg_filename = f'avg_time_{num_satellites}_satellites.npy'
        np.save(avg_filename, AvgPerN)


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


    # After simulation, call the plotting function
    
    avg_filename = f'avg_time_satellites.npy'

    np.save(avg_filename, AvgPerN)

    ### Falk ###
    time_steps = [entry['time_step'] for entry in gdop_val]
    gdop_values = [entry['min_gdop'] for entry in gdop_val]

    # Create a plot
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
    plt.close()  # Close after saving

    # Plot Earth, user, and satellites
    plot_earth_and_satellites(user_position, best_combination, satellite_positions, visible_satellites)
    ### Falk ###

    # Falk | GDOP map and plot
    #gdop_map, satellite_count_map, num_lat, num_lon = calculate_gdop_for_grid(satellite_positions, num_lat=number_latitude, num_lon=number_longitude)
    #plot_gdop_map(gdop_map)
    #plot_satellites_over_latitude(satellite_count_map)

    print(f"Maximum GDOP: {max_gdop}")
    print(f"Minimum GDOP: {min_gdop}")
    print(f"Average GDOP: {avg_gdop}")

    #plot_3d_distribution(gdop_map, num_lat, num_lon)
