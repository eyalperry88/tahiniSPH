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

Tahini phase shifts was simulated using [PySPH](https://pysph.readthedocs.io/en/latest/), a very powerful framework for fluid simulation. The system contains two types of particles: tahini and solid (bowl, spoon). The tahini particles contain a property which corresponds to the amount of H2O aorund that particle. In addition to to fluid flow equations (See Implementation) - a new equation was added to the system, each two particles interact by Van der Waalce force (Lennard Jones potential) dependent on a Gaussian of the sum of the H2O for both particles.

## Implementation

The code is based on [hydrostatic_tank](https://github.com/pypr/pysph/blob/master/pysph/examples/hydrostatic_tank.py) example from PySPH, where fluid particles are floating in a tank. The examples contains three ways to check for the boundary between the solid and liquid particles:

 - Adami et al. "A generalized wall boundary condition for smoothed
   particle hydrodynamics", 2012, JCP, 231, pp 7057--7075 (REF1)

 - Monaghan and Kajtar, "SPH particle boundary forces for arbitrary
   boundaries", 2009, 180, pp 1811--1820 (REF2)

 - Gesteria et al. "State-of-the-art of classical SPH for free-surface
   flows", 2010, JHR, pp 6--27 (REF3)
   
REF2 requires special spacing between fluid and solid particles, and REF1 proved to be 75% slower in my experiments, so I'm using the third formulation (the attached code supports REF1 & REF2)


