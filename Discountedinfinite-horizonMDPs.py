import numpy as np
from typing import Dict, Tuple, Callable, List
from scipy.stats import rv_discrete

class InfiniteHorizonMDP:
    def __init__(self, S: int, A: int, P: np.ndarray, r: np.ndarray, gamma: float):
        """
        Initialize a discounted infinite-horizon MDP.
        
        Args:
            S: Number of states (states are indexed 0 to S-1)
            A: Number of actions (actions are indexed 0 to A-1)
            P: Transition probability tensor of shape (S, A, S)
            r: Reward function array of shape (S, A)
            gamma: Discount factor (0 <= gamma < 1)
        """
        self.S = S
        self.A = A
        self.P = P
        self.r = r
        self.gamma = gamma
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate the MDP parameters."""
        assert 0 <= self.gamma < 1, "Discount factor must be in [0,1)"
        assert self.P.shape == (self.S, self.A, self.S), "Invalid transition probability shape"
        assert self.r.shape == (self.S, self.A), "Invalid reward function shape"
        
        # Check transition probabilities sum to 1 for each (s,a)
        for s in range(self.S):
            for a in range(self.A):
                assert np.isclose(self.P[s,a].sum(), 1), f"Transition probabilities don't sum to 1 for (s={s},a={a})"
                assert (self.P[s,a] >= 0).all(), f"Negative transition probability for (s={s},a={a})"
        
        # Check rewards are in [0,1]
        assert (self.r >= 0).all() and (self.r <= 1).all(), "Rewards must be in [0,1]"
    
    def value_function(self, policy: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        Compute the value function V^π for a given policy using iterative policy evaluation.
        
        Args:
            policy: Stochastic policy array of shape (S, A) where policy[s,a] = π(a|s)
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            Value function array of shape (S,)
        """
        V = np.zeros(self.S)
        
        for _ in range(max_iter):
            V_new = np.zeros(self.S)
            for s in range(self.S):
                # Compute expected immediate reward
                expected_reward = np.sum(policy[s] * self.r[s])
                
                # Compute expected future value
                expected_future_value = 0
                for a in range(self.A):
                    for s_next in range(self.S):
                        expected_future_value += policy[s,a] * self.P[s,a,s_next] * V[s_next]
                
                V_new[s] = expected_reward + self.gamma * expected_future_value
            
            if np.max(np.abs(V - V_new)) < tol:
                break
                
            V = V_new
            
        return V
    
    def q_function(self, policy: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        Compute the Q-function Q^π for a given policy.
        
        Args:
            policy: Stochastic policy array of shape (S, A) where policy[s,a] = π(a|s)
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            Q-function array of shape (S, A)
        """
        Q = np.zeros((self.S, self.A))
        
        for _ in range(max_iter):
            Q_new = np.zeros((self.S, self.A))
            for s in range(self.S):
                for a in range(self.A):
                    # Immediate reward
                    Q_new[s,a] = self.r[s,a]
                    
                    # Expected future value
                    future_value = 0
                    for s_next in range(self.S):
                        # For next state s_next, take expectation over actions according to policy
                        future_value += self.P[s,a,s_next] * np.sum(policy[s_next] * Q[s_next])
                    
                    Q_new[s,a] += self.gamma * future_value
            
            if np.max(np.abs(Q - Q_new)) < tol:
                break
                
            Q = Q_new
            
        return Q
    
    def optimal_value_function(self, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        Compute the optimal value function V* using value iteration.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            Optimal value function array of shape (S,)
        """
        V = np.zeros(self.S)
        
        for _ in range(max_iter):
            V_new = np.zeros(self.S)
            for s in range(self.S):
                # For each action, compute Q(s,a)
                q_values = np.zeros(self.A)
                for a in range(self.A):
                    q_values[a] = self.r[s,a] + self.gamma * np.sum(self.P[s,a] * V)
                
                # Take maximum over actions
                V_new[s] = np.max(q_values)
            
            if np.max(np.abs(V - V_new)) < tol:
                break
                
            V = V_new
            
        return V
    
    def optimal_q_function(self, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        Compute the optimal Q-function Q* using Q-value iteration.
        
        Args:
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            Optimal Q-function array of shape (S, A)
        """
        Q = np.zeros((self.S, self.A))
        
        for _ in range(max_iter):
            Q_new = np.zeros((self.S, self.A))
            for s in range(self.S):
                for a in range(self.A):
                    # Immediate reward
                    Q_new[s,a] = self.r[s,a]
                    
                    # Maximum future value
                    future_value = 0
                    for s_next in range(self.S):
                        future_value += self.P[s,a,s_next] * np.max(Q[s_next])
                    
                    Q_new[s,a] += self.gamma * future_value
            
            if np.max(np.abs(Q - Q_new)) < tol:
                break
                
            Q = Q_new
            
        return Q
    
    def get_optimal_policy(self, Q: np.ndarray = None) -> np.ndarray:
        """
        Get the optimal deterministic policy from the optimal Q-function.
        
        Args:
            Q: Optional Q-function (if None, computes optimal Q-function)
            
        Returns:
            Deterministic policy array of shape (S,) where policy[s] is the optimal action
        """
        if Q is None:
            Q = self.optimal_q_function()
            
        return np.argmax(Q, axis=1)
    
    def discounted_occupancy(self, policy: np.ndarray, rho: np.ndarray, max_iter: int = 1000, tol: float = 1e-6) -> np.ndarray:
        """
        Compute the discounted occupancy distribution d^π_ρ.
        
        Args:
            policy: Stochastic policy array of shape (S, A)
            rho: Initial state distribution of shape (S,)
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            
        Returns:
            Discounted occupancy distribution array of shape (S, A)
        """
        d = np.zeros((self.S, self.A))
        
        # Initialize state distribution with rho
        state_dist = rho.copy()
        
        for t in range(max_iter):
            d_new = np.zeros((self.S, self.A))
            
            # For each state-action pair
            for s in range(self.S):
                for a in range(self.A):
                    if t == 0:
                        # At t=0, depends only on initial state distribution
                        d_new[s,a] = (1 - self.gamma) * rho[s] * policy[s,a]
                    else:
                        # For t>0, accumulate from previous state-action pairs
                        for s_prev in range(self.S):
                            for a_prev in range(self.A):
                                d_new[s,a] += self.gamma * d[s_prev,a_prev] * self.P[s_prev,a_prev,s] * policy[s,a]
            
            if np.max(np.abs(d - d_new)) < tol and t > 0:
                break
                
            d = d_new
            
        return d
    
    def weighted_value_function(self, policy: np.ndarray, rho: np.ndarray) -> float:
        """
        Compute the weighted value function V^π(ρ).
        
        Args:
            policy: Stochastic policy array of shape (S, A)
            rho: State distribution of shape (S,)
            
        Returns:
            Weighted value function value
        """
        V = self.value_function(policy)
        return np.sum(rho * V)
