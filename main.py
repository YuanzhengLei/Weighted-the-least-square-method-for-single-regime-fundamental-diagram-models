import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
plt.rcParams['font.family'] = 'Times New Roman'

x = []
y = []
# Load data from Excel file
with open('GA400.txt', 'r') as file:
    for line in file:
        # Split the line into parts
        parts = line.split()
        # Append data to respective lists
        if len(parts) >= 3:
            x.append(float(parts[1])/1.609344)
            y.append(float(parts[2])/1.609344)

X = np.array(x).reshape(-1, 1)
Y = np.array(y).reshape(-1, 1)

sorted_indices = np.argsort(X, axis=0).flatten()
X = X[sorted_indices]
Y = Y[sorted_indices]

weights = []

r'''
for u in range(len(X)):
    if u == 0:
        matching_indices = [i for i, x in enumerate(X) if x == X[0]]
        max_index = max(matching_indices)
        if max_index == 0:
            weights.append(X[max_index + 1][0] - X[0][0])
        else:
            weights.append((X[max_index + 1][0] - X[0][0])/max_index)
    if u == len(X) - 1:
        weights.append(X[u][0] - X[u - 1][0])
    if u >= 1 and u < len(X) - 1:
        matching_indices = [i for i, x in enumerate(X) if x == X[u]]
        max_index = max(matching_indices)
        weights.append((X[max_index + 1][0] - X[u - 1][0])/2 * (max_index - u + 1))


df = pd.DataFrame({'Weights': weights})
df.to_csv("Weights.csv", index=False)
r'''


df1 = pd.read_csv('Weights.csv')
weights = df1['Weights'].values
Weights = np.array(weights).reshape(-1, 1)



def loss_function_Greenshields(params):
    v_max, p_max = params
    predicted_y = v_max * (1 - X / p_max)
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50]

# Run optimization
result = minimize(loss_function_Greenshields, initial_guess)

v_max_estimated, p_max_estimated = result.x


print("Greenshields Model")
print(f"Estimated v_max: {v_max_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * (1 - X / p_max_estimated), color='navy', label='Greenshields Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Greenshields Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Greenshields.pdf')
plt.show()

def loss_function_Greenberg(params):
    v_critical, p_max = params
    predicted_y = v_critical * np.log(p_max/X)
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50]

# Run optimization
result = minimize(loss_function_Greenberg, initial_guess)

v_critical_estimated, p_max_estimated = result.x


print("Greenberg Model")
print(f"Estimated v_critical: {v_max_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_critical_estimated * np.log(p_max_estimated/X), color='navy', label='Greenberg Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Greenberg Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Greenberg.pdf')
plt.show()

def loss_function_Underwood(params):
    v_max, p_critical = params
    predicted_y = v_max * np.exp(-X/p_critical)
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50]

# Run optimization
result = minimize(loss_function_Underwood, initial_guess)

v_max_estimated, p_critical_estimated = result.x


print("Underwood Model")
print(f"Estimated v_max_: {v_max_estimated:.4f}")
print(f"Estimated p_critical: {p_max_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * np.exp(-X/p_critical_estimated), color='navy', label='Underwood Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Underwood Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Underwood.pdf')
plt.show()


def loss_function_Newell(params):
    v_max, p_max, lambda_para = params
    predicted_y = v_max * (1 - np.exp((-lambda_para/v_max)*(1/X - 1/p_max)))
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50, 10]

# Run optimization
result = minimize(loss_function_Newell, initial_guess)

v_max_estimated, p_max_estimated, lambda_para_estimated = result.x


print("Newell Model")
print(f"Estimated v_max_: {v_max_estimated:.4f}")
print(f"Estimated p_critical: {p_critical_estimated:.4f}")
print(f"Estimated lambda_para: {lambda_para_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * np.exp(-X/p_critical_estimated), color='navy', label='Newell Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Underwood Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Newell.pdf')
plt.show()

def loss_function_Pipes(params):
    v_max, p_max, n = params
    ratio = 1 - X / p_max
    predicted_y = np.where(ratio < 0, 0, v_max * ratio ** n)
    return np.sum((Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [100, 50, 10]

# Run optimization
result = minimize(loss_function_Pipes, initial_guess)

v_max_estimated, p_max_estimated, n_estimated = result.x


print("Pipes Model")
print(f"Estimated v_max_: {v_max_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")
print(f"Estimated n: {n_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * np.exp(-X/p_max_estimated), color='navy', label='Pipes Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Pipes Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Pipes.pdf')
plt.show()


def loss_function_Papageogiou(params):
    v_max, p_max, alpha_para = params
    if alpha_para <= 0:
        return np.inf
    if p_max <= 0:
        return np.inf
    try:
        predicted_y = v_max * np.exp(- (1 / alpha_para) * (X / p_max) ** alpha_para)
    except FloatingPointError:
        return np.inf
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50, 10]

# Run optimization
result = minimize(loss_function_Papageogiou, initial_guess)

v_max_estimated, p_max_estimated, alpha_para_estimated = result.x


print("Papageogiou Model")
print(f"Estimated v_max_: {v_max_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")
print(f"Estimated alpha_para: {alpha_para_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * np.exp(- (1 / alpha_para_estimated) * (X / p_max_estimated) ** alpha_para_estimated), color='navy', label='Papageogiou Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Papageogiou Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Papageogiou.pdf')
plt.show()

def loss_function_Kerner(params):
    v_max, p_critical = params
    predicted_y = v_max * (1/(1 + np.exp(((X/p_critical)-0.25)/(0.06)) - 3.72 * 10 ** -6))
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50]

# Run optimization
result = minimize(loss_function_Kerner, initial_guess)

v_max_estimated, p_critical_estimated = result.x


print("Kerner Model")
print(f"Estimated v_max_: {v_max_estimated:.4f}")
print(f"Estimated p_critical: {p_critical_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * (1/(1 + np.exp(((X/p_critical_estimated)-0.25)/(0.06)) - 3.72 * 10 ** -6)), color='navy', label='Kerner Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Kerner Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Kerner.pdf')
plt.show()


def loss_function_DelCastillo_Benitez_a(params):
    v_max, p_max, P_w = params
    predicted_y = v_max * ( 1 -np.exp((P_w/v_max)*(1 - p_max/X)))
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50, 10]

# Run optimization
result = minimize(loss_function_DelCastillo_Benitez_a, initial_guess)

v_max_estimated, p_max_estimated, P_w_estimated = result.x


print("DelCastillo_Benitez_a Model")
print(f"Estimated v_max_: {v_max_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")
print(f"Estimated P_w: {P_w_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * ( 1 -np.exp((P_w_estimated/v_max_estimated)*(1 - p_max_estimated/X))), color='navy', label='DelCastillo_Benitez_a Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and DelCastillo_Benitez_a Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('DelCastillo_Benitez_a.pdf')
plt.show()


def loss_function_Jayakrishnan(params):
    v_min, v_max, p_max = params
    predicted_y = v_min + (v_max - v_min) * (1 - X/p_max)
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [0, 50, 50]

# Run optimization
result = minimize(loss_function_Jayakrishnan, initial_guess)

v_min_estimated, v_max_estimated, p_max_estimated = result.x


print("Jayakrishnan Model")
print(f"Estimated v_min_: {v_min_estimated:.4f}")
print(f"Estimated v_max: {v_max_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_min_estimated + (v_max_estimated - v_min_estimated) * (1 - X/p_max_estimated), color='navy', label='Jayakrishnan Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Jayakrishnan Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Jayakrishnan.pdf')
plt.show()


def loss_function_ArdekaniandGhandehari(params):
    v_critical, p_min, p_max = params
    predicted_y = v_critical * np.log((p_max + p_min)/(X + p_min))
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 0, 50]

# Run optimization
result = minimize(loss_function_ArdekaniandGhandehari, initial_guess)

v_critical_estimated, p_min_estimated, p_max_estimated = result.x


print("ArdekaniandGhandehari Model")
print(f"Estimated v_critical: {v_critical_estimated:.4f}")
print(f"Estimated p_min: {p_min_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_critical_estimated * np.log((p_max_estimated + p_min_estimated)/(X + p_min_estimated)), color='navy', label='ArdekaniandGhandehari Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and ArdekaniandGhandehari Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('ArdekaniandGhandehari.pdf')
plt.show()


def loss_function_Lee(params):
    v_max, p_max, E, gamma = params
    predicted_y = v_max * (1 - X/p_max) * (1 + E * (X/p_max) ** gamma) ** (-1)
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50, 10, 10]

# Run optimization
result = minimize(loss_function_Lee, initial_guess)

v_max_estimated, p_max_estimated, E_estimated, gamma_estimated = result.x


print("Lee Model")
print(f"Estimated v_critical: {v_critical_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")
print(f"Estimated E: {E_estimated:.4f}")
print(f"Estimated gamma: {gamma_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * (1 - X/p_max_estimated) * (1 + E_estimated * (X/p_max_estimated) ** gamma_estimated) ** (-1), color='navy', label='Lee Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Lee Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Lee.pdf')
plt.show()


def loss_function_Drake(params):
    v_max, p_critical = params
    predicted_y = v_max * np.power(np.exp( -(X/p_critical)), 2)
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [50, 50]

# Run optimization
result = minimize(loss_function_Drake, initial_guess)

v_max_estimated, p_critical_estimated = result.x


print("Drake Model")
print(f"Estimated v_max: {v_max_estimated:.4f}")
print(f"Estimated p_critical: {p_critical_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * np.power(np.exp( -(X/p_critical_estimated)), 2), color='navy', label='Drake Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Drake Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Drake.pdf')
plt.show()

def loss_function_MacNicholas(params):
    v_max, p_max, n, m = params
    numerator = np.power(p_max, n) - np.power(X, n)
    denominator = np.power(p_max, n) + m * np.power(X, n)
    predicted_y = v_max * (numerator / denominator)
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [90.0, 50.0, 10.0, 10.0]

# Run optimization
result = minimize(loss_function_MacNicholas, initial_guess)

v_max_estimated, p_max_estimated, n_estimated, m_estimated = result.x
numerator_estimated = np.power(p_max_estimated, n_estimated) - np.power(X, n_estimated)
denominator_estimated = np.power(p_max_estimated, n_estimated) + m_estimated * np.power(X, n_estimated)

print("MacNicholas Model")
print(f"Estimated v_max: {v_max_estimated:.4f}")
print(f"Estimated p_max: {p_max_estimated:.4f}")
print(f"Estimated n: {n_estimated:.4f}")
print(f"Estimated m: {m_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated * (numerator_estimated / denominator_estimated), color='navy', label='MacNicholas Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and MacNicholas Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('MacNicholas.pdf')
plt.show()


def loss_function_Wang(params):
    v_max, v_critical, p_critical, theta_1, theta_2 = params
    exp_component = np.exp((X - p_critical) / theta_1)
    logistic_function = 1 / (1 + exp_component)
    predicted_y = v_critical + ((v_max - v_critical) * np.power(logistic_function, theta_2))
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [90.0, 10.0, 10.0, 10.0, 10.0]

# Run optimization
result = minimize(loss_function_Wang, initial_guess)

v_max_estimated, v_critical_estimated, p_critical_estimated, theta_1_estimated, theta_2_estimated = result.x
exp_component_estimated = np.exp((X - p_critical_estimated) / theta_1_estimated)
logistic_function_estimated = 1 / (1 + exp_component_estimated)

print("Wang Model")
print(f"Estimated v_max: {v_max_estimated:.4f}")
print(f"Estimated v_critical: {v_critical_estimated:.4f}")
print(f"Estimated p_critical: {p_critical_estimated:.4f}")
print(f"Estimated theta_1: {theta_1_estimated:.4f}")
print(f"Estimated theta_2: {theta_2_estimated:.4f}")

# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_critical_estimated + ((v_max_estimated - v_critical_estimated) * np.power(logistic_function_estimated, theta_2_estimated)), color='navy', label='Wang Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Wang Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Wang.pdf')
plt.show()


def loss_function_Cheng(params):
    v_max, p_critical, m = params
    term = np.power(X / p_critical, m)
    predicted_y = v_max / np.power(1 + term, 2 / m)
    return np.sum(Weights * (Y - predicted_y) ** 2)

# Initial guesses for parameters
initial_guess = [90.0, 10.0, 10.0]

# Run optimization
result = minimize(loss_function_Cheng, initial_guess)

v_max_estimated, p_critical_estimated, m_estimated = result.x
term_estimated = np.power(X / p_critical_estimated, m_estimated)

print("Cheng Model")
print(f"Estimated v_max: {v_max_estimated:.4f}")
print(f"Estimated p_critical: {p_critical_estimated:.4f}")
print(f"Estimated m: {m_estimated:.4f}")


# Plot the raw data and the fitted model
plt.scatter(X, Y, label="Raw data", color='red', s=1)
plt.plot(X, v_max_estimated / np.power(1 + term_estimated, 2 / m_estimated), color='navy', label='Cheng Model')
plt.ylabel('V(mph)')
plt.xlabel('P(Veh/mile)')
plt.title('Raw Data and Cheng Model')
plt.legend()
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.savefig('Cheng.pdf')
plt.show()