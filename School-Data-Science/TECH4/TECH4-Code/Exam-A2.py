import pyomo.environ as pyo
solver_name = 'appsi_highs'
SOLVER = pyo.SolverFactory(solver_name)
assert SOLVER.available(), f'Solver {solver_name} is not available'
import math

model = pyo.ConcreteModel("The northernmost")

# Specify number of rows and columns in the grid.
rows = 10
cols = 10

# Define parameter for fixed cost of installing an electrical station at all coordiantes (r, c).
F = [[2000 for c in range(cols)] for r in range(rows)]
F[4][4] = 2500
F[4][5] = 2500
F[5][4] = 2500
F[5][5] = 2500

# Define a parameter N containing the coordinates (r, c) of all j neighbourhoods in the grid.
N = [
     (0, 3), 
     (2, 2), (2, 4), (2, 6), (2, 7), 
     (3, 7), (3, 8), 
     (4, 2), (4, 4), 
     (5, 1), (5, 5), (5,7), 
     (6, 8), 
     (7, 4), (7, 9), 
     (8, 1), (8, 3), (8, 4), (8, 6), 
     (9, 5)
     ]

# Define a nested dictionary D with two levels: 
# The outer keys are the coordinates (r, c) of neighbourhood j and the inner keys are all possible coordinates (r, c) in the grid where a station can be installed.
# The values in dictionary are the distances between neighbourhood j and all coordinates (r, c), calculated using the formula provided in the task.
D = dict() 
for j in N: # For each neighbourhood j in N...
    D[j] = {(r, c) : math.sqrt((j[0] - r)**2  + (j[1] - c)**2) for r in range(rows) for c in range(cols)} # Create 1 entry in the dictionary containing a dictionary with the distance to all (r, c) coordinates in the grid.

# Variable cost of installing 1 km of electrical wires.
V = 100

# Decision variables:

model.s = pyo.Var(range(rows), range(cols), domain=pyo.Binary)      # s is a binary variable that equals 1 if an electrical station is installed at coordinate (r, c).
model.w = pyo.Var(N, range(rows), range(cols), domain=pyo.Binary)   # w is a binary variable that equals 1 if neighbourhood j is connected to electrical station at coordinates (r, c).

# Objective function:

model.objective = pyo.Objective(
    expr = sum(F[r][c] * model.s[r, c] for r, c in model.s.keys()) +        # Sums the fixed costs of the installed electrical stations
    V * sum(D[j][(r, c)] * model.w[j, r, c] for j in D for r, c in D[j]),   # Sums the distance between each neighbourhood and electrical stations, and multiplies with variable cost per km.
    sense = pyo.minimize
)

# Initialize constraintlist:
model.constraints = pyo.ConstraintList()

# 20 constraints ensuring that every neighbourhood j is connected to 1 station.
for j in N:
    model.constraints.add(sum(model.w[j, r, c] for r in range(rows) for c in range(cols)) == 1)

# 2000 constraints ensuring that neighbourhood j can only be connected to a station at coordinates (r, c) if it exists.
for j in N:
    for r, c in model.s.keys():
        model.constraints.add((model.w[j, r, c] <= model.s[r, c]))

SOLVER.solve(model)

# This section is for displaying results:

# Create a dictionary with the installed electrical stations as keys, and all neighbourhoods covered by them as values.
covered = {}
for j in N:
    for r in range(rows):
        for c in range(cols):
            val = model.w[j, r, c].value
            if val > 0:
                if (r, c) not in covered:
                    covered[(r, c)] = []
                covered[(r, c)].append(j)

# The total number of neighbourhoods covered by the installed electrical stations.
covered_neighbourhoods = sum(len(covered[station]) for station in covered.keys())

# Print results:
print(f'\nSolution to the {model.name} problem | Task A2')
print(f'\nNumber of neighbourhoods covered: {covered_neighbourhoods}')
print(f'Number of electrical station(s) installed: {len(covered)}')
print(f'\nThe optimal coordinates for the station(s) are: {[station for station in covered.keys()]}, (0-indexed)')

for loc in covered:
    print(f'\nThe following neighbourhoods is covered by an electrical station at coordinates {loc}')
    print(covered[loc])

print(f'\nThe total cost of covering {covered_neighbourhoods} neighbourhoods: $ {pyo.value(model.objective):.2f}\n')