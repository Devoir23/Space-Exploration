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