import pulp

model = pulp.LpProblem('SandBox', pulp.LpMaximize)

var_a = pulp.LpVariable(f"p_a", lowBound=0)
var_b = pulp.LpVariable(f"p_b", lowBound=0)
var_c = pulp.LpVariable(f"p_c", lowBound=0)
var_ab = pulp.LpVariable(f"p_ab", lowBound=0)
var_ac = pulp.LpVariable(f"p_ac", lowBound=0)
var_bc = pulp.LpVariable(f"p_bc", lowBound=0)
var_abc = pulp.LpVariable(f"p_abc", lowBound=0)

# UM constraints.
model += var_a + var_b + var_ab >= 3
model += var_b + var_c + var_bc >= 3
model += var_a + var_c + var_ac >= 3
model += var_a + var_b + var_c + var_ab + var_ac + var_bc + var_abc <= 4

model.solve(pulp.PULP_CBC_CMD(msg=False))

# value = pulp.value(model.objective)
status = pulp.LpStatus[model.status]

# print(f"value = {value}, status = {status}")
print(f"status = {status}")
print(var_a.varValue)
print(var_b.varValue)
print(var_c.varValue)
print(var_ab.varValue)
print(var_ac.varValue)
print(var_bc.varValue)
print(var_abc.varValue)
