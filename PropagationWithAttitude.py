import math
import random




class Satellite:
    def __init__(self, num_satellites_per_plane, num_planes,i, orbit=None, epsys=None, commsys=None, DataHand=None, PropSys=None, Optic=None, category=None, flag=None):
        # Sun-synchronous parameters
        self.processing_time = 0  # Add the processing_time attribute

        semimajoraxis = 26571 #20200 # Galileo 2nd paper: 29602 # Galileo23222 # F:7070  # Corresponding to 700-800 km altitudes
        true_anomaly_increment = 360 / num_satellites_per_plane
        raan_increment = 360 / num_planes

        # Calculate inclination for SSO
        #cos_i = (-2 * math.pi * semimajoraxis_sso**3 * n_eq / (J2 * omega_E * R_Earth**2))**(1/7)
        
        inclination = 56 # Galileo 56 # F:45  # Convert from radians to degrees
        #inclination=98.2
        #print(inclination)
        
        if orbit is None:
            # orbit = {
            #     'semimajoraxis': semimajoraxis,  # km
            #     'inclination': inclination,  # degrees
            #     'eccentricity': 0,
            #     'raan': 360 / num_planes * (i % num_planes),
            #     #'raan': 0,
            #     'arg_of_perigee':0,
            #     'true_anomaly': 360 / num_satellites_per_plane * (i % num_satellites_per_plane),
            #     #'true_anomaly': 0,

            #     # add more orbital parameters as needed
            #     # add more orbital parameters as needed
            # }
            orbit = {
                'semimajoraxis': semimajoraxis,  # km
                'inclination': inclination,  # degrees
                'eccentricity': 0,
                'raan': raan_increment * (i // num_satellites_per_plane),  # Divide by num_satellites_per_plane to distribute evenly across planes
                'arg_of_perigee': 0,
                'true_anomaly': true_anomaly_increment * (i % num_satellites_per_plane),
            }
        #Electric Power Subsystem
        
        if epsys is None: 
            epsys = {
                'EnergyStorage': 84*3600,
                'SolarPanelSize': 0.4*0.3,
                # add more subsystems as needed
            }

        # Comm Subsystem

        if commsys is None:
            commsys= {
                #Random communication system on board
                # 1: UHF
                # 2: VHF
                # 3: S-band
                # 4: X-band
                'band': 5
            }

        #Data Handling

        if DataHand is None:
            DataHand = {
                'DataStorage': 8*64e9, # Maximum storage onboard. 8*[bytes]=[bites], from ISISpace bus
            }
        
        # Propulsion System

        if PropSys is None:
            PropSys = {
                'PropellantMass': 1, # Maximum propellant onboard. [kg]
                'PropulsionType': 0,  #This value is a flag to define if we have chemical or electrical propulsion
                'SpecificImpulse': 250, #Specific impulse of the propulsion system in [s]
                'Thrust': 1, # Thrust of the propulsion system in [N]
            }

        # Optical Payload 
        
        if Optic is None:
            Optic = {
                'ApertureDiameter': 0.09, # Aperture diameter [m]
                'Wavelength': 700e-9, #Max wavelength of observation
            }
        
        # Index for define if the satellite is an observer or a target or something else
        if category is None:
            category = {
                'Target': 0, # Index for target to be identified
                'Observation': 1, #Index for observation satellite
            }


   
# =============================================================================
#         if flag is None:
#             flag = random.choice([True, False])
# =============================================================================
        self.orbit = orbit
        self.epsys = epsys
        self.commsys = commsys
        self.DataHand = DataHand
        self.PropSys = PropSys
        self.Optic = Optic
        self.category = category
        self.flag = flag
        
    def propagate(self, time_step):
        # propagate the orbit of the satellite using the Euler method
        # time_step: the time step in seconds
        a = self.orbit['semimajoraxis'] * 1000  # the semimajor axis in meters
        e = self.orbit['eccentricity']
        i = math.radians(self.orbit['inclination'])
        omega = math.radians(self.orbit['arg_of_perigee'])
        Omega = math.radians(self.orbit['raan'])
        theta = math.radians(self.orbit['true_anomaly'])

        # calculate the mean motion of the satellite in radians per second
        n = math.sqrt(3.986004418e14 / a ** 3)  # the mean motion in radians per second

        # calculate the eccentric anomaly using Kepler's equation
        E = math.atan2(math.sqrt(1 - e ** 2) * math.sin(theta), e + math.cos(theta))
        M = E - e * math.sin(E)

        # calculate the true anomaly and the radius vector
        E_dot = n / (1 - e * math.cos(E))
        theta_dot = math.sqrt(1 - e ** 2) * E_dot
        theta += theta_dot * time_step
        r = a * (1 - e ** 2) / (1 + e * math.cos(theta))

        # calculate the position and velocity vectors of the satellite in the inertial frame
        x = r * (math.cos(Omega) * math.cos(omega + theta) - math.sin(Omega) * math.sin(omega + theta) * math.cos(i))
        y = r * (math.sin(Omega) * math.cos(omega + theta) + math.cos(Omega) * math.sin(omega + theta) * math.cos(i))
        z = r * math.sin(omega + theta) * math.sin(i)
        v = (-n * r / math.sqrt(1 - e ** 2) * (
                    math.cos(Omega) * math.sin(omega + theta) + math.sin(Omega) * math.cos(omega + theta) * math.cos(
                i)),
             n * r / math.sqrt(1 - e ** 2) * (math.sin(Omega) * math.sin(omega + theta) - math.cos(Omega) * math.cos(
                 omega + theta) * math.cos(i)),
             n * r / math.sqrt(1 - e ** 2) * math.sin(i))

        # update the orbit parameters

        # update the orbit parameters
        self.orbit['true_anomaly'] = math.degrees(theta)
        self.orbit['mean_anomaly'] = math.degrees(M)
        self.orbit['radius'] = r / 1000
        self.orbit['x'], self.orbit['y'], self.orbit['z'] = x, y, z
        self.orbit['vx'], self.orbit['vy'], self.orbit['vz'] = v
        # print(v)

        return (x, y, z), v  # return the position and velocity vectors of the satellite

class TargetSatellite(Satellite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = "target"
        # Any other specific attributes/methods for the target satellite

class ObserverSatellite(Satellite):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.category = "observer"
        # Any other specific attributes/methods for the observer satellite


def distance_between(sat1, sat2, time_step):
    # propagate both satellites to the same time
    pos1, vel1 = sat1.propagate(time_step)
    pos2, vel2 = sat2.propagate(time_step)

    # calculate the distance between the two satellites
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    # print(distance)
    return distance
