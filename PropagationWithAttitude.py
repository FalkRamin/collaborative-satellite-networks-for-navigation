import math

class Satellite:
    def __init__(self, num_satellites_per_plane, num_planes,i, orbit=None, flag=None):
        self.processing_time = 0  # Add the processing_time attribute

        true_anomaly_increment = 360 / num_satellites_per_plane
        raan_increment = 360 / num_planes

        # Defaults (units: SMA in km, angles in deg)
        defaults = {
            'semimajoraxis': 26571.0,       # km
            'inclination': 56.0,            # degrees
            'eccentricity': 0.0,
            'raan': raan_increment * (i // num_satellites_per_plane), # Divide by num_satellites_per_plane to distribute evenly across planes
            'arg_of_perigee': 0.0,
            'true_anomaly': true_anomaly_increment * (i % num_satellites_per_plane),
        }

        # Merge defaults with the input orbit template
        merged_orbit = defaults.copy()
        if orbit:
            merged_orbit.update(orbit)

        self.orbit = merged_orbit
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
