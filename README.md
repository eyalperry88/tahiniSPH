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

The code is based on [hydrostatic_tank](https://github.com/pypr/pysph/blob/master/pysph/examples/hydrostatic_tank.py) example from PySPH, where fluid particles are floating in a tank. Each PySPH program basically contains three parts: particles, equations and a solver. I will describe the 2D version, although little changes are required to move to 3D (or higher!)

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

Next, we define properties for each particle. The volume and mass are computed as in the hydrostatic_tank example:

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

and we add our special tahini-water property. 

### Define the Equations

The examples contains three ways to check for the boundary between the solid and liquid particles:

 - Adami et al. "A generalized wall boundary condition for smoothed
   particle hydrodynamics", 2012, JCP, 231, pp 7057--7075 (REF1)

 - Monaghan and Kajtar, "SPH particle boundary forces for arbitrary
   boundaries", 2009, 180, pp 1811--1820 (REF2)

 - Gesteria et al. "State-of-the-art of classical SPH for free-surface
   flows", 2010, JHR, pp 6--27 (REF3)
   
REF2 requires special spacing between fluid and solid particles, and REF1 proved to be 75% slower in my experiments, so I'm using the third formulation (the attached code supports REF1 & REF2)

### Solve it
