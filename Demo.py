"""
This is a demonstration of how the package works
"""
from scipy.stats import norm

from src.simulation.simulation_model import SimulationModel, describe

data = norm(loc=0, scale=1).rvs(size=100000)
simulation = SimulationModel(data=data)
sample = simulation.sample(num_simulation=100)
print(sample)
print("----------------------")
print(describe(sample))
print("----------------------")