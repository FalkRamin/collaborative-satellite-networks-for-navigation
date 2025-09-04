\# Satellite GDOP Simulation



This project simulates a Walker Delta constellation and calculates GDOP values for a specific receiver position or a GDOP map.



Change the input parameters according to the interested scenario.



The mask angle describes the minimum elevation angle of the local horizon of the user, so that it is able to see a satellite. 



The GDOP calculation works like the following:

1. Check if the receiver is able to see 4 or more satellites
2. Calculate for each combination of 4 visible satellites a GDOP value
3. Save the set of 4 satellites in that time step that yield the best GDOP value



Output files the simulation produces:

1. gdop\_output.txt 
2. visible\_satellites\_output.txt
3. Additional graphs set individually in the main(e.g., GDOP vs time)



The gdop\_output.txt contains, for each timestep, the GDOP value, and the Cartesian coordinates of the 4 satellites that were selected as the best combination yielding this GDOP value.



The visible\_satellite\_output.txt contains, for each timestep, all the Cartesian coordinates of all visible satellites. 



