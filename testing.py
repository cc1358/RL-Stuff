# Example: 3-state, 2-action MDP
S = 3
A = 2
gamma = 0.9

# Transition probabilities: P[s,a,s']
P = np.array([
    [[0.7, 0.2, 0.1],  # State 0
     [0.1, 0.8, 0.1]],  # Action 0 and 1 for state 0
    
    [[0.3, 0.4, 0.3],   # State 1
     [0.0, 0.9, 0.1]],
    
    [[0.5, 0.5, 0.0],   # State 2
     [0.2, 0.2, 0.6]]
])

# Rewards: r[s,a]
r = np.array([
    [0.5, 0.8],  # State 0
    [0.1, 0.9],  # State 1
    [0.7, 0.3]   # State 2
])

# Create MDP
mdp = InfiniteHorizonMDP(S, A, P, r, gamma)

# Random policy: choose actions uniformly
random_policy = np.ones((S, A)) / A

# Compute value function for random policy
V_pi = mdp.value_function(random_policy)
print("Value function for random policy:", V_pi)

# Compute Q-function for random policy
Q_pi = mdp.q_function(random_policy)
print("Q-function for random policy:", Q_pi)

# Compute optimal value function
V_star = mdp.optimal_value_function()
print("Optimal value function:", V_star)

# Compute optimal Q-function
Q_star = mdp.optimal_q_function()
print("Optimal Q-function:", Q_star)

# Get optimal policy
optimal_policy = mdp.get_optimal_policy()
print("Optimal policy (deterministic):", optimal_policy)

# Compute discounted occupancy for random policy with uniform initial distribution
rho = np.ones(S) / S
d_pi = mdp.discounted_occupancy(random_policy, rho)
print("Discounted occupancy for random policy:", d_pi)

# Compute weighted value function
V_pi_rho = mdp.weighted_value_function(random_policy, rho)
print("Weighted value function for random policy:", V_pi_rho)
