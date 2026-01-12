import pyomo.environ as pyo
solver_name = 'appsi_highs'
SOLVER = pyo.SolverFactory(solver_name)
assert SOLVER.available(), f'Solver {solver_name} is not available'
import math

model = pyo.ConcreteModel("The northernmost")

rows = 10
cols = 10

F = [[2000 for c in range(cols)] for r in range(rows)]
F[4][4] = 2500
F[4][5] = 2500
F[5][4] = 2500
F[5][5] = 2500

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

D = dict() 
for j in N: 
    D[j] = {(r, c) : math.sqrt((j[0] - r)**2  + (j[1] - c)**2) for r in range(rows) for c in range(cols)} 

V = 100

model.s = pyo.Var(range(rows), range(cols), domain=pyo.Binary)      
model.w = pyo.Var(N, range(rows), range(cols), domain=pyo.Binary)   

model.objective = pyo.Objective(
    expr = sum(F[r][c] * model.s[r, c] for r, c in model.s.keys()) +  
    V * sum(D[j][(r, c)] * model.w[j, r, c] for j in D for r, c in D[j]), 
    sense = pyo.minimize
)

model.constraints = pyo.ConstraintList()

# For this problem, we modify this constraint to allow neighbourhoods to be connected to 1 or less stations.
for j in N:
    model.constraints.add(sum(model.w[j, r, c] for r in range(rows) for c in range(cols)) <= 1)

for j in N:
    for r, c in model.s.keys():
        model.constraints.add((model.w[j, r, c] <= model.s[r, c]))

# We also need to add this constraint that ensures that atleast 14 neighbourhoods are connected on the grid.
model.constraints.add(sum(model.w[j, r, c] for r in range(rows) for c in range(cols) for j in N) >= 14)

SOLVER.solve(model)

covered = {}
for j in N:
    for r in range(rows):
        for c in range(cols):
            val = model.w[j, r, c].value
            if val > 0:
                if (r, c) not in covered:
                    covered[(r, c)] = []
                covered[(r, c)].append(j)

covered_neighbourhoods = sum(len(covered[station]) for station in covered.keys())

print(f'\nSolution to the {model.name} problem | Task A4')
print(f'\nNumber of neighbourhoods covered: {covered_neighbourhoods}')
print(f'Number of electrical station(s) installed: {len(covered)}')
print(f'\nThe optimal coordinates for the station(s) are: {[station for station in covered.keys()]}, (0-indexed)')

for loc in covered:
    print(f'\nThe following neighbourhoods is covered by an electrical station at coordinates {loc}')
    print(covered[loc])

print(f'\nThe total cost of covering {covered_neighbourhoods} neighbourhoods: $ {pyo.value(model.objective):.2f}\n')