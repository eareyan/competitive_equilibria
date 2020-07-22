import pulp

model = pulp.LpProblem('SandBox', pulp.LpMaximize)

var_p_0 = pulp.LpVariable(f"p_0", lowBound=0)
var_p_1 = pulp.LpVariable(f"p_1", lowBound=0)
var_p_2 = pulp.LpVariable(f"p_2", lowBound=0)

# UM constraints.
model += 0 >= 5 - var_p_0 - var_p_2
model += 0 >= 8 - var_p_0 - var_p_1
model += 10 - var_p_1 - var_p_2 >= 0

# Revenue constraints
model += var_p_1 + var_p_2 >= 0.0
model += var_p_1 + var_p_2 >= var_p_0
model += var_p_1 + var_p_2 >= var_p_1
model += var_p_1 + var_p_2 >= var_p_2
model += var_p_1 + var_p_2 >= var_p_0 + var_p_1
model += var_p_1 + var_p_2 >= var_p_0 + var_p_2
model += var_p_1 + var_p_2 >= var_p_1 + var_p_2
model += var_p_1 + var_p_2 >= var_p_0 + var_p_1 + var_p_2

model.solve(pulp.PULP_CBC_CMD(msg=False))

# value = pulp.value(model.objective)
status = pulp.LpStatus[model.status]

# print(f"value = {value}, status = {status}")
print(f"status = {status}")
print(var_p_0.varValue)
print(var_p_1.varValue)
print(var_p_2.varValue)
