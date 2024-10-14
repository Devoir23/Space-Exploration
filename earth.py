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


# todo: Euler method Integration
def euler_method(r, v, accn, dt):
    for i in range(1, len(t)):
        r[i] = r[i-1] + v[i-1]*dt
        v[i] = v[i-1] + accn(r[i-1])*dt


# todo: Apply Euler method
euler_method(r,v,acceleration,dt)


# todo: RK4 method integration
def rk4_method(r, v, acceleration, dt):
    """
    ODE for position:
    --> dr/dt = v
    --> r_new = r_old + dt/6(k1r + 2*k2r + 2*k3r + k4)

    ODE for velocity:
    --> dv/dt = a
    --> v_new = v_old + dt/6(k1v + 2*k2v + 2*k3r + k4)

    step 1: 0
    k1v = accb(r[i-1]);  k1r = v[i-1]

    step 2: dt/2 using step 1
    k2v = accn(r[i-1]+1 + k1r * dt/2 ); k2r = v[i-1] + k1v * dt/2

    step 3:
    k3v = accn(r[i-1] + k2r * dt/2), k3r = v[i-1] + k2v * dt/2
    step 4:
    k4v
    k4r

    """

    for i in range(1,len(r)):
        k1v = acceleration(r[i - 1])
        k1r = v[i - 1]

        k2v = acceleration(r[i - 1] + k1r * dt / 2)
        k2r = v[i - 1] + k1v * dt / 2

        k3v = acceleration(r[i - 1] + k2r * dt / 2)
        k3r = v[i - 1] + k2v * dt / 2

        k4v = acceleration(r[i - 1] + k3r * dt )
        k4r = v[i - 1] + k3v * dt

        # update the r and v
        v[i] = v[i-1] + dt/6*(k1v + 2*k2v + 2*k3v + k4v)
        r[i] = r[i-1] + dt/6*(k1r + 2*k2r + 2*k3r + k4r)

# rk4_method(r, v, acceleration, dt)


def numerical_integration(r, v, acceleration, dt, method="euler"):
    if method.lower()=="euler":
        euler_method(r, v, acceleration, dt)
    elif method.lower()=="rk4":
        rk4_method(r, v, acceleration, dt)
    else:
        raise Exception(f'You can either choose "euler" or "rk4". Your current input for methodd is : {method}')

# call numerical integration function
numerical_integration(r, v, acceleration, dt, method="euler")

# todo: position and velocity of earth at Aphelion
sizes = np.array([np.linalg.norm(position) for position in r])
position_aphelion = np.max(sizes)
arg_aphelion = np.argmax(sizes)
velocity_aphelion = np.linalg.norm(v[arg_aphelion])

print(position_aphelion/10e8, velocity_aphelion/10e2)