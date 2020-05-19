"""Hydrostatic tank example. (2 minutes)

This example is from (Section 6.0) of Adami et. al. JCP 231, 7057-7075.

This is a good problem to test the implementation of the wall boundary
condition. Physically, a column of fluid is left in an open tank and
allowed to settle to equilibrium. Upon settling, a linear pressure
field (p = rho*g*h) should be established according to elementary
fluid mechanics.

Different boundary formulations can be used to check for this behaviour:

 - Adami et al. "A generalized wall boundary condition for smoothed
   particle hydrodynamics", 2012, JCP, 231, pp 7057--7075 (REF1)

 - Monaghan and Kajtar, "SPH particle boundary forces for arbitrary
   boundaries", 2009, 180, pp 1811--1820 (REF2)

 - Gesteria et al. "State-of-the-art of classical SPH for free-surface
   flows", 2010, JHR, pp 6--27 (REF3)

Of these, the first and third are ghost particle methods while the
second is the classical Monaghan style, repulsive particle approach.

For the fluid dynamics, we use the multi-phase formulation presented
in REF1.

ffmpeg -framerate 25 -start_number 53 -i frame%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" output.mp4

"""

import os.path

import numpy as np

# PyZoltan imports
from cyarray.api import LongArray

from compyle.api import declare


# PySPH imports
from pysph.base.utils import get_particle_array_wcsph as gpa
from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.kernels import Gaussian, WendlandQuintic, CubicSpline, QuinticSpline
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.sph.integrator import PECIntegrator, EPECIntegrator
from pysph.sph.integrator_step import WCSPHStep

# the eqations
from pysph.sph.equation import Group, Equation

# Equations for REF1
from pysph.sph.wc.transport_velocity import VolumeFromMassDensity,\
    ContinuityEquation,\
    MomentumEquationPressureGradient, \
    MomentumEquationArtificialViscosity,\
    SolidWallPressureBC

# Monaghan type repulsive boundary forces used in REF(2)
from pysph.sph.boundary_equations import MonaghanBoundaryForce,\
    MonaghanKajtarBoundaryForce

# Equations for the standard WCSPH formulation and dynamic boundary
# conditions defined in REF3
from pysph.sph.wc.basic import TaitEOS, TaitEOSHGCorrection, MomentumEquation
from pysph.sph.basic_equations import XSPHCorrection, \
    MonaghanArtificialViscosity, SummationDensity

# Equations for RigidBodies
from pysph.sph.rigid_body import BodyForce, RigidBodyMoments, RigidBodyMotion, \
    RK2StepRigidBody, RigidBodyCollision, AkinciRigidFluidCoupling, \
    SummationDensityBoundary

# domain and reference values
Lx = 2.0
Ly = 1.0
Lz = 2.0
Cx = 1.0
Cy = 1.0
Cz = 1.0
bowlR = 1.0
tahiniH = 0.5
H = 0.9
gy = -1.0
Vmax = np.sqrt(abs(gy) * H)
c0 = 10 * Vmax
rho0 = 1000.0
spoon_rho=500
p0 = c0 * c0 * rho0
gamma = 1.0

# Reynolds number and kinematic viscosity
Re = 100
nu = Vmax * Ly / Re

# Numerical setup
nx = 50
dx = Lx / nx
ghost_extent = 5.5 * dx
hdx = 1.2

# adaptive time steps
h0 = hdx * dx
dt_cfl = 0.25 * h0 / (c0 + Vmax)
dt_viscous = 0.125 * h0**2 / nu
dt_force = 0.25 * np.sqrt(h0 / abs(gy))

tdamp = 1.0
tf = 2.0
dt = 0.75 * min(dt_cfl, dt_viscous, dt_force)
# output_at_times = np.arange(0.1, 2.1, 0.1)

#  simple harmonic oscillator motion
class HarmonicOscilllator(Equation):
    def __init__(self, dest, sources, A=4.0, omega=0.5):
        self.A = A
        self.omega = omega
        super(HarmonicOscilllator, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_u, d_av, d_aw, d_w, t):
        d_u[d_idx] = self.A * self.omega * cos(self.omega * 2 * M_PI * t)
        d_au[d_idx] = (self.A * self.omega) ** 2 * cos(self.omega * 2 * M_PI * t) * (cos(self.omega * 2 * M_PI * t) - 2 * sin(self.omega * 2 * M_PI * t))
        d_w[d_idx] = self.A * self.omega * sin(self.omega * 2 * M_PI * t)
        d_aw[d_idx] = (self.A * self.omega) ** 2 * sin(self.omega * 2 * M_PI * t) * (sin(self.omega * 2 * M_PI * t) - 2 * cos(self.omega * 2 * M_PI * t))

# a Lennard-Jones/Van der Waals force dependent on a gaussian distributon of H2O per particle
class TahiniEquation(Equation):
    def __init__(self, dest, sources, sigma):
        # Overload this only if you need to pass additional constants
        # Otherwise, no need to override __init__
        self.eps = 0.5      # magic number
        self.sigma = sigma  # the distance in which particles have no effect
        self.var = 12       # the inverse variance of the gaussian. we want this so e^(-1 * var) ~ 0
        super(TahiniEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_m, d_au, d_av, d_aw, d_h2o_amount, s_idx, s_m, s_h2o_amount, RIJ, XIJ):
        if RIJ > 1e-9:
            # Gaussian distrbution for tahini-water-tahini interaction
            p = M_E ** (- (d_h2o_amount[d_idx] + s_h2o_amount[s_idx] - 1) ** 2 * self.var)

            # Forced derived from Lennard-Jones potential
            # V_LJ = 4 * self.eps * ((self.sigma ** 12 / RIJ ** 12) - (self.sigma ** 6 / RIJ ** 6))
            F_LJ = 24 * self.eps * (- 2 * (self.sigma ** 12 / RIJ ** 13) + (self.sigma ** 6 / RIJ ** 7))

            # normal vector passing from particle i to j
            nij_x = -XIJ[0] / RIJ
            nij_y = -XIJ[1] / RIJ
            nij_z = -XIJ[2] / RIJ
        else:
            p = 0.0
            F_LJ = 0.0

            nij_x = 0.0
            nij_y = 0.0
            nij_z = 0.0

        d_au[d_idx] += p * F_LJ * nij_x
        d_av[d_idx] += p * F_LJ * nij_y
        d_aw[d_idx] += p * F_LJ * nij_z

class H2OFaucet(Equation):
    """Applies a "faucet" - constant refill of H2O for a specific subset of particles"""
    def __init__(self, dest, sources, x, y, z, r, fill_rate, lag=2):
        self.faucet_x = x
        self.faucet_y = y
        self.faucet_z = z
        self.faucet_r2 = r  ** 2
        self.faucet_fill_rate = fill_rate
        self.omega = 0.5
        self.lag = lag
        super(H2OFaucet, self).__init__(dest, sources)

    def initialize(self, d_idx, d_h2o_amount, d_x, d_y, d_z, t, dt):
        if t > self.lag and ((d_x[d_idx] - self.faucet_x) ** 2 + (d_y[d_idx] - self.faucet_y) ** 2 + (d_z[d_idx] - self.faucet_z) ** 2) < self.faucet_r2:
            d_h2o_amount[d_idx] += self.faucet_fill_rate * dt * (cos(self.omega * 2 * M_PI * t) ** 2)

class DiffuseH2O(Equation):
    """Diffusion of H2O between particles

    Based on: https://github.com/AlexanderPuckhaber/FluidSimulationThesis/wiki/PySPH---Diffusion
    """
    def __init__(self, dest, sources, diffusion_speed):
        self.diffusion_speed = diffusion_speed
        super(DiffuseH2O, self).__init__(dest, sources)

    def initialize(self, d_idx, d_h2o_velocity):
        d_h2o_velocity[d_idx] = 0.0

    def loop(self, d_idx, s_idx, d_h2o_velocity, s_h2o_velocity, d_h2o_amount, s_h2o_amount, WIJ):
        h2o_gradient = (s_h2o_amount[s_idx] - d_h2o_amount[d_idx])
        d_h2o_velocity[d_idx] += ( h2o_gradient * self.diffusion_speed ) * WIJ
        s_h2o_velocity[s_idx] -= ( h2o_gradient * self.diffusion_speed ) * WIJ

    def post_loop(self, d_idx, d_h2o_amount, d_h2o_velocity, dt, t):
        d_h2o_amount[d_idx] += d_h2o_velocity[d_idx] * dt

def create_particles_xy():
    _x = np.arange(-ghost_extent, Lx + ghost_extent, dx)
    _y = np.arange(-ghost_extent, Ly, dx)
    _z = np.arange(-ghost_extent, Lz + ghost_extent, dx)
    x, y, z = np.meshgrid(_x, _y, _z)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    # filter out anything that is not the bowl
    p_bowl = []
    for i in range(x.size):
        r = (x[i] - Cx)**2 + (y[i] - Cy)**2 + (z[i] - Cz)**2
        if r > bowlR**2 and r < (bowlR + ghost_extent)**2 and y[i] < 0.95:
            p_bowl.append(i)

    print("Num points in bowl:", len(p_bowl))

    x_bowl = x[p_bowl]
    y_bowl = y[p_bowl]
    z_bowl = z[p_bowl]

    p_spoon = []
    for i in range(x.size):
        if y[i] > 0.25 and (((x[i] - Cx)**2 + (y[i] - 0.4)**2 < (0.15**2)) or ((x[i] - Cx) ** 2 < 0.04**2)) and (z[i] - Cz + 0.5) ** 2 < (0.04 ** 2):
            p_spoon.append(i)

    print("Num points in spoon:", len(p_spoon))
    x_spoon = x[p_spoon]
    y_spoon = y[p_spoon]
    z_spoon = z[p_spoon]

    p_spoon = set(p_spoon)

    # tahini fills the bowl
    p = []
    for i in range(x.size):
        if i in p_spoon:
            continue
        r = (x[i] - Cx)**2 + (y[i] - Cy)**2 + (z[i] - Cz)**2
        if r < bowlR**2 and y[i] < 0.5:
            p.append(i)

    x_tahini = x[p]
    y_tahini = y[p]
    z_tahini = z[p]

    return x_bowl, y_bowl, z_bowl, x_spoon, y_spoon, z_spoon, x_tahini, y_tahini, z_tahini


def visualize_t0():
    import matplotlib.pyplot as plt
    # please run this function to know how
    # geometry looks like
    x_bowl, y_bowl, z_bowl, x_spoon, y_spoon, z_spoon, x_tahini, y_tahini, z_tahini = create_particles_xy()
    plt.scatter(x_bowl, y_bowl)
    plt.scatter(x_tahini, y_tahini)
    plt.scatter(x_spoon, y_spoon)
    plt.axes().set_aspect('equal', 'datalim')
    print("done")
    plt.show()


class TahiniBowl(Application):
    def add_user_options(self, group):
        group.add_argument(
            '--bc-type', action='store', type=int,
            dest='bc_type', default=1,
            help="Specify the implementation type one of (1, 2, 3)"
        )

    def create_particles(self):
        # create the particle arrays
        x_bowl, y_bowl, z_bowl, x_spoon, y_spoon, z_spoon, x_tahini, y_tahini, z_tahini = create_particles_xy()

        bowl = gpa(name='bowl', x=x_bowl, y=y_bowl, z=z_bowl)

        tahini = gpa(name='tahini', x=x_tahini, y=y_tahini, z=z_tahini)

        spoon = gpa(name='spoon', x=x_spoon, y=y_spoon, z=z_spoon)

        print("Tahini bowl :: ntahini = %d, nbowl=%d, dt = %g" % (
            tahini.get_number_of_particles(),
            bowl.get_number_of_particles(), dt))

        ###### ADD PARTICLE PROPS FOR MULTI-PHASE SPH ######

        # particle volume
        tahini.add_property('V')
        bowl.add_property('V')
        spoon.add_property('V')

        # kernel sum term for boundary particles
        bowl.add_property('wij')
        spoon.add_property('wij')

        # advection velocities and accelerations
        for name in ('auhat', 'avhat', 'awhat'):
            tahini.add_property(name)

        ##### INITIALIZE PARTICLE PROPS #####
        tahini.rho[:] = rho0
        bowl.rho[:] = rho0
        spoon.rho[:] = rho0

        # mass is set to get the reference density of rho0
        volume = dx * dx * dx

        # volume is set as dx^2
        tahini.V[:] = 1. / volume
        bowl.V[:] = 1. / volume
        spoon.V[:] = 1. / volume

        tahini.m[:] = volume * rho0
        bowl.m[:] = volume * rho0
        spoon.m[:] = volume * rho0 #* 1e-3

        # smoothing lengths
        tahini.h[:] = hdx * dx
        bowl.h[:] = hdx * dx
        spoon.h[:] = hdx * dx

        ##### INITIALIZE TAHINI/WATER PROPS #####
        tahini.add_property('h2o_amount')
        #tahini.h2o_amount[:] = np.random.normal(0.5, 0.05, n)
        tahini.add_property('h2o_velocity')

        # lean and mean
        tahini.set_output_arrays(['h2o_amount', 'x', 'y', 'z', 'u', 'w', 'v'])
        bowl.set_output_arrays(['gid', 'x', 'y', 'z', 'u', 'w', 'v'])
        spoon.set_output_arrays(['gid', 'x', 'y', 'z', 'u', 'w', 'v'])

        # return the particle list
        return [tahini, bowl, spoon]

    def create_solver(self):
        # Create the kernel
        #kernel = Gaussian(dim=3)
        kernel = CubicSpline(dim=3)
        #kernel = QuinticSpline(dim=3)

        integrator = PECIntegrator(tahini=WCSPHStep(), spoon=WCSPHStep())

        # Create a solver.
        solver = Solver(kernel=kernel, dim=3, integrator=integrator,
                        tf=tf, dt=dt,
                        adaptive_timestep=False
                        )
        return solver

    def create_equations(self):
        # Formulation for REF1
        equations1 = [
            Group(equations=[
                HarmonicOscilllator(dest='spoon', sources=None, A=3, omega=0.5),

                # Translate acceleration to positions
                XSPHCorrection(dest='spoon', sources=['spoon'], eps=0.0)
            ], real=False),

            # Water Faucet Equations
            Group(equations=[
                H2OFaucet(dest='tahini', sources=None, x=0.5, y=tahiniH, z=1, r=0.1, fill_rate=8),
                DiffuseH2O(dest='tahini', sources=['tahini'], diffusion_speed=0.025),
            ]),

            # For the multi-phase formulation, we require an estimate of the
            # particle volume. This can be either defined from the particle
            # number density or simply as the ratio of mass to density.
            Group(equations=[
                VolumeFromMassDensity(dest='tahini', sources=None)
            ], ),

            # Equation of state is typically the Tait EOS with a suitable
            # exponent gamma
            Group(equations=[
                TaitEOSHGCorrection(
                    dest='tahini',
                    sources=None,
                    rho0=rho0,
                    c0=c0,
                    gamma=gamma),
            ], ),

            # The boundary conditions are imposed by extrapolating the tahini
            # pressure, taking into considering the bounday acceleration
            Group(equations=[
                SolidWallPressureBC(dest='bowl', sources=['tahini'], b=1.0, gy=gy,
                                    rho0=rho0, p0=p0),
                SolidWallPressureBC(dest='spoon', sources=['tahini'], b=1.0, gy=gy,
                                    rho0=rho0, p0=p0),
            ], ),

            # Main acceleration block
            Group(equations=[
                TahiniEquation(dest='tahini', sources=['tahini'], sigma=dx / 1.122),


                # Continuity equation
                ContinuityEquation(
                    dest='tahini', sources=[
                        'tahini', 'bowl', 'spoon']),

                # Pressure gradient with acceleration damping.
                MomentumEquationPressureGradient(
                    dest='tahini', sources=['tahini', 'bowl', 'spoon'], pb=0.0, gy=gy,
                    tdamp=tdamp),

                # artificial viscosity for stability
                MomentumEquationArtificialViscosity(
                    dest='tahini', sources=['tahini', 'bowl', 'spoon'], alpha=1, c0=c0),

                # Position step with XSPH
                XSPHCorrection(dest='tahini', sources=['tahini'], eps=0.0)
            ]),
        ]

        # Formulation for REF3
        equations3 = [
            # Spoon Equations
            Group(equations=[
                HarmonicOscilllator(dest='spoon', sources=None, A=3, omega=0.333),

                # Translate acceleration to positions
                XSPHCorrection(dest='spoon', sources=['spoon'], eps=0.0)
            ], real=False),

            # Water Faucet Equations
            Group(equations=[
                H2OFaucet(dest='tahini', sources=None, x=0.5, y=tahiniH, z=1, r=0.1, fill_rate=25),
                DiffuseH2O(dest='tahini', sources=['tahini'], diffusion_speed=0.025),
            ]),

            # For the multi-phase formulation, we require an estimate of the
            # particle volume. This can be either defined from the particle
            # number density or simply as the ratio of mass to density.
            Group(equations=[
                VolumeFromMassDensity(dest='tahini', sources=None)
            ], ),

            # Equation of state is typically the Tait EOS with a suitable
            # exponent gamma. The solid phase is treated just as a fluid and
            # the pressure and density operations is updated for this as well.
            Group(equations=[
                TaitEOS(
                    dest='tahini',
                    sources=None,
                    rho0=rho0,
                    c0=c0,
                    gamma=gamma),
                TaitEOS(
                    dest='bowl',
                    sources=None,
                    rho0=rho0,
                    c0=c0,
                    gamma=gamma),
                TaitEOS(
                    dest='spoon',
                    sources=None,
                    rho0=rho0,
                    c0=c0,
                    gamma=gamma),
            ], ),

            # Main acceleration block. The boundary conditions are imposed by
            # peforming the continuity equation and gradient of pressure
            # calculation on the bowl phase, taking contributions from the
            # tahini phase
            Group(equations=[
                TahiniEquation(dest='tahini', sources=['tahini'], sigma=dx / 1.122),

                # Continuity equation
                ContinuityEquation(
                    dest='tahini', sources=[
                        'tahini', 'bowl', 'spoon']),
                ContinuityEquation(dest='bowl', sources=['tahini']),
                ContinuityEquation(dest='spoon', sources=['tahini']),

                # Pressure gradient with acceleration damping.
                MomentumEquationPressureGradient(
                    dest='tahini', sources=['tahini', 'bowl', 'spoon'], pb=0.0, gy=gy,
                    tdamp=tdamp),

                # artificial viscosity for stability
                MomentumEquationArtificialViscosity(
                    dest='tahini', sources=['tahini', 'bowl', 'spoon'], alpha=1, c0=c0),

                # Position step with XSPH
                XSPHCorrection(dest='tahini', sources=['tahini'], eps=0.5)

            ]),
        ]

        if self.options.bc_type == 1:
            return equations1
        elif self.options.bc_type == 3:
            return equations3


if __name__ == '__main__':
    #visualize_t0()
    app = TahiniBowl()
    app.run()
