import pulp

model = pulp.LpProblem('SandBox', pulp.LpMaximize)

the_low_bound = -0.2
# the_low_bound = -math.inf
var_a = pulp.LpVariable(f"p_a", lowBound=the_low_bound)
var_b = pulp.LpVariable(f"p_b", lowBound=the_low_bound)
var_c = pulp.LpVariable(f"p_c", lowBound=the_low_bound)
var_ab = pulp.LpVariable(f"p_ab", lowBound=the_low_bound)
var_ac = pulp.LpVariable(f"p_ac", lowBound=the_low_bound)
var_bc = pulp.LpVariable(f"p_bc", lowBound=the_low_bound)
var_abc = pulp.LpVariable(f"p_abc", lowBound=the_low_bound)

# Revenue-maximization objective.
model += pulp.lpSum([var_a + var_b + var_c + var_ac + var_ab + var_bc + var_abc])

# UM constraints: each consumer should be happy with its allocation.
model += var_a + var_b + var_ab >= 3
model += var_b + var_c + var_bc >= 3
model += var_a + var_c + var_ac >= 3
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc <= 4

# Revenue constraints: the seller should maximize revenue allocating everything.
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc >= 0
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc >= var_a
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc >= var_b
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc >= var_c
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc >= var_ab + var_a + var_b
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc >= var_ac + var_a + var_c
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc >= var_bc + var_b + var_c

model.solve(pulp.PULP_CBC_CMD(msg=False))

value = pulp.value(model.objective)
status = pulp.LpStatus[model.status]

print(f"status = {status}, value = {value}")
print(f"a = {var_a.varValue:.4f}")
print(f"b = {var_b.varValue:.4f}")
print(f"c = {var_c.varValue:.4f}")
print(f"ab = {var_ab.varValue:.4f}")
print(f"ac = {var_ac.varValue:.4f}")
print(f"bc = {var_bc.varValue:.4f}")
print(f"abc = {var_abc.varValue:.4f}")

# Bundle prices

# Singleton bundle prices.
print(f"P(a) = {var_a.varValue}")
print(f"P(b) = {var_b.varValue}")
print(f"P(c) = {var_c.varValue}")

# Quadratic bundle prices.
print(f"P(ab) = {var_a.varValue + var_b.varValue + var_ab.varValue}")
print(f"P(ac) = {var_a.varValue + var_c.varValue + var_ac.varValue}")
print(f"P(bc) = {var_b.varValue + var_c.varValue + var_bc.varValue}")

# Cubic bundle prices.
print(f"P(abc) = {var_a.varValue + var_b.varValue + var_c.varValue + var_ab.varValue + var_ac.varValue + var_bc.varValue + var_abc.varValue}")
