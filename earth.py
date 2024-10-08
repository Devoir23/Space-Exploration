# Imports
import matplotlib.pyplot as plt
import numpy
import numpy as np
# Constants
G = 6.6743e-11
M_sun = 1.9891e30 # mass of sun in kg

# Initial Position and Velocity
r_0 = np.array([147.1e9, 0]) # m
v_0 = np.array([0, -30.29e3])  # m/s

# Time steps and total time for simulation
dt = 3600 # sec
t_max = 3.154e7 # sec

# Time array to be used in numerical solution
t = np.arange(0, t_max, dt)
# print(t.astype('int32'))

# initialize arrays to store positions and velocities at all the time steps
r = np.empty(shape=(len(t), 2))
v = np.empty(shape=(len(t), 2))

# set the initial conditions for position and velocity
r[0], v[0] = r_0, v_0


# Acceleration function: when passed in thw position vector
def acceleration(r):
    return (-G*M_sun / np.linalg.norm(r)**3) * r

# print(acceleration(r_0)) --> output: [-0.00613532 -0.        ]


# Euler method Integration
def euler_method(r, v, accn, dt):
    for i in range(1, len(t)):
        r[i] = r[i-1] + v[i-1]*dt
        v[i] = v[i-1] + accn(r[i-1])*dt


# Apply Euler method
euler_method(r,v,acceleration,dt)

# position and velocity of earth at Aphelion
sizes = np.array([np.linalg.norm(position) for position in r])
position_aphelion = np.max(sizes)
arg_aphelion = np.argmax(sizes)
velocity_aphelion = np.linalg.norm(v[arg_aphelion])

print(position_aphelion/10e8, velocity_aphelion/10e2)