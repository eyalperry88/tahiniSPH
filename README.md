## Background

Tahini is amazing. A condiment made from toasted ground hulled sesame, tahini can be added to a variety of dishes and drinks to improve taste, texture and [nutritional value](https://fdc.nal.usda.gov/fdc-app.html#/food-details/168604/nutrients).

Tahini is best when purchased as 100% sesame paste.
![100% Tahini Products](assets/tahini_products.png)

To prepare it, water, lemon and favorite spices are slowly added and stirred. In this process, something very odd happens. The tahini at room temperature starts as a viscous fluid. As water is added to the mixture, the tahini goes through a *phase shift* and becomes granular solid. As more water is added, the tahini returns to a fluid and delicious state.

Video demonstration:

![Tahini phase shifts]()

Notice that while sharing certain similarities, Tahini is *not* a "classical" non-Newtonian fluid, which exhibits phase shifts as response to stress. Tahini's phase is defined the amount of water in the system, and the solid phase is dependent on "just the right amount" - too little or too much water, and tahini is still liquid (viscous, but liquid). Not exhibited in the video, having the right amount for a long time creates such a solid piece of Tahini that it might break a blender. Be careful.

This process is sometimes described as "seizing" and is not unique to Tahini. Peanut butter, as well as non edible materials such as plaster, are known to seize up water and solidify before going to liquid state. This happens as small amounts water act as a cross-linker between molecules. We assume a similar process happens for tahini, while the liquid starting point is due to the high oil concentration.

## Simulation

![Full 3D simulation video]()

Tahini phase shifts was simulated using [PySPH](https://pysph.readthedocs.io/en/latest/), a powerful framework for fluid simulation. The system contains two types of particles: tahini and solid (bowl, spoon). The tahini particles contain a property which corresponds to the amount of H2O aorund that particle. In addition to to fluid flow equations (See Implementation) - a new equation was added to the system, each two particles interact by Van der Waalce force (Lennard Jones potential) dependent on a Gaussian of the sum of the H2O for both particles.

## Implementation

The code is based on [hydrostatic_tank](https://github.com/pypr/pysph/blob/master/pysph/examples/hydrostatic_tank.py) example from PySPH, where fluid particles are floating in a tank. Each PySPH program basically contains three parts: particles, equations and a solver. First, I will describe the 2D version of fluid siumlation in a bowl mixed by a spoon (no phase shifts, yet)

### Create the Particles

We create a bowl, tahini and a mixing spoon. We create a uniform mesh grid and pick the indices relevant for each object:

```python
# dimensions
Lx = 2.0
Ly = 1.0
Cx = 1.0
Cy = 1.0
bowlR = 1.0
tahiniH = 0.5

# the higher this number, the more refined the simulation is (and the longer it takes...)
nx = 100
dx = Lx / nx
ghost_extent = 5.5 * dx

def create_particles_xy():
    _x = np.arange(-ghost_extent, Lx + ghost_extent, dx)
    _y = np.arange(-ghost_extent, Ly, dx)
    x, y = np.meshgrid(_x, _y)
    x = x.ravel()
    y = y.ravel()

    # filter out anything that is not the bowl
    p_bowl = []
    for i in range(x.size):
        r = (x[i] - Cx)**2 + (y[i] - Cy)**2
        if r > bowlR**2 and r < (bowlR + ghost_extent)**2:
            p_bowl.append(i)

    x_bowl = x[p_bowl]
    y_bowl = y[p_bowl]

    # spoon has a special shape
    p_spoon = []
    for i in range(x.size):
        if y[i] > 0.25 and ((x[i] > 0.6 and x[i] < 0.8 and y[i] < tahiniH) or (x[i] > 0.66 and x[i] < 0.74)):
            p_spoon.append(i)

    x_spoon = x[p_spoon]
    y_spoon = y[p_spoon]

    # tahini fills the bowl
    p = []
    for i in range(x.size):
        if i in p_spoon:
            continue
        r = (x[i] - Cx)**2 + (y[i] - Cy)**2
        if r < bowlR**2 and y[i] < tahiniH:
            p.append(i)

    x_tahini = x[p]
    y_tahini = y[p]

    return x_bowl, y_bowl, x_spoon, y_spoon,x_tahini, y_tahini
```

Next, we define properties for each particle:

```python
tahini.rho[:] = rho0
bowl.rho[:] = rho0
spoon.rho[:] = rho0

# mass is set to get the reference density of rho0
volume = dx * dx

# volume is set as dx^2
tahini.V[:] = 1. / volume
bowl.V[:] = 1. / volume
spoon.V[:] = 1. / volume

tahini.m[:] = volume * rho0
bowl.m[:] = volume * rho0
spoon.m[:] = volume * rho0
```

### Define the Equations

The hydrostatic_tank contains three ways to check for the boundary between the solid and liquid particles:

 - [Adami et al. "A generalized wall boundary condition for smoothed
   particle hydrodynamics", 2012, JCP, 231, pp 7057--7075](https://www.sciencedirect.com/science/article/pii/S002199911200229X?casa_token=aQ5VRuReYuEAAAAA:5HpBoFwsU_cXWDH4BeM7h2iShXCEA-rcXqHn4GO5diZAe3NfHG3Qh76-iSlcG-C2t4YYmqSiEA) (REF1)

 - [Monaghan and Kajtar, "SPH particle boundary forces for arbitrary
   boundaries", 2009, 180, pp 1811--1820](https://ui.adsabs.harvard.edu/abs/2009CoPhC.180.1811M/abstract) (REF2)

 - Gesteria et al. "State-of-the-art of classical SPH for free-surface
   flows", 2010, JHR, pp 6--27 (REF3)

REF2 requires special spacing between fluid and solid particles, and REF1 proved to be 75% slower in my experiments, so I'm using the third formulation:

```python
equations3 = [
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
        # Continuity equation
        ContinuityEquation(dest='tahini', sources=['tahini', 'bowl', 'spoon']),
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
 ```

 Each of these equations define a set of confitions that will be integrated over the 'dest' particles. In case of particle-particles interactions, the sources parameter defines each particle type

 A few basic equations (from Adami 2012):
 - ContinuityEquation - conservation of mass
$$
\frac{d\rho_a}{dt} = \rho_a \sum_b \frac{m_b}{\rho_b} \boldsymbol{v}_{ab} \cdot \nabla_a W_{ab
$$

- MomentumEquationPressureGradient - pressure
$$
\frac{d \boldsymbol{v}_a}{dt} = \frac{1}{m_a}\sum_b (V_a^2 +V_b^2)\left[-\bar{p}_{ab}\nabla_a W_{ab} \right]
$$

- MomentumEquationArtificialViscosity - viscosity

$
\frac{d \boldsymbol{v}_a}{dt} = -\sum_b m_b \alpha h_{ab}
        c_{ab} \frac{\boldsymbol{v}_{ab}\cdot
        \boldsymbol{r}_{ab}}{\rho_{ab}\left(|r_{ab}|^2 + \epsilon
        \right)}\nabla_a W_{ab}
$


Ok what about now

$ H = ax $

### Solve it
