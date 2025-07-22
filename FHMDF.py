import numpy as np
from typing import List, Tuple

class FiniteHorizonMDP:
    def __init__(self, S: int, A: int, H: int, P: List[np.ndarray], r: List[np.ndarray]):
        """
        Initialize a finite-horizon MDP.
        
        Args:
            S: Number of states (states are indexed 0 to S-1)
            A: Number of actions (actions are indexed 0 to A-1)
            H: Horizon length
            P: List of transition probability tensors, each of shape (S, A, S)
            r: List of reward function arrays, each of shape (S, A)
        """
        self.S = S
        self.A = A
        self.H = H
        self.P = P
        self.r = r
        
        # Validate inputs
        self._validate_inputs()
        
    def _validate_inputs(self):
        """Validate the MDP parameters."""
        assert len(self.P) == self.H, "Need transition probabilities for each time step"
        assert len(self.r) == self.H, "Need reward functions for each time step"
        
        for h in range(self.H):
            assert self.P[h].shape == (self.S, self.A, self.S), f"Invalid transition probability shape at step {h}"
            assert self.r[h].shape == (self.S, self.A), f"Invalid reward function shape at step {h}"
            
            # Check transition probabilities sum to 1 for each (s,a)
            for s in range(self.S):
                for a in range(self.A):
                    assert np.isclose(self.P[h][s,a].sum(), 1), f"Transition probabilities don't sum to 1 for (h={h},s={s},a={a})"
                    assert (self.P[h][s,a] >= 0).all(), f"Negative transition probability for (h={h},s={s},a={a})"
            
            # Check rewards are in [0,1]
            assert (self.r[h] >= 0).all() and (self.r[h] <= 1).all(), f"Rewards must be in [0,1] at step {h}"
    
    def value_function(self, policy: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute the value function V^π for a given policy using backward induction.
        
        Args:
            policy: List of deterministic policies where policy[h][s] gives action at step h
            
        Returns:
            List of value function arrays V[h][s] for each time step h
        """
        # Initialize V[H+1] = 0
        V = [np.zeros(self.S) for _ in range(self.H+1)]
        
        # Backward induction
        for h in range(self.H-1, -1, -1):
            for s in range(self.S):
                a = policy[h][s]
                # Immediate reward
                V[h][s] = self.r[h][s,a]
                # Expected future value
                for s_next in range(self.S):
                    V[h][s] += self.P[h][s,a,s_next] * V[h+1][s_next]
        
        return V[:-1]  # Exclude V[H] which is all zeros
    
    def q_function(self, policy: List[np.ndarray]) -> List[np.ndarray]:
        """
        Compute the Q-function Q^π for a given policy using backward induction.
        
        Args:
            policy: List of deterministic policies where policy[h][s] gives action at step h
            
        Returns:
            List of Q-function arrays Q[h][s,a] for each time step h
        """
        # Initialize Q[H] = r[H] since V[H+1] = 0
        Q = [np.zeros((self.S, self.A)) for _ in range(self.H+1)]
        
        # Backward induction
        for h in range(self.H-1, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    # Immediate reward
                    Q[h][s,a] = self.r[h][s,a]
                    # Expected future value
                    for s_next in range(self.S):
                        # Next action is determined by policy at h+1
                        a_next = policy[h+1][s_next]
                        Q[h][s,a] += self.P[h][s,a,s_next] * Q[h+1][s_next,a_next]
        
        return Q[:-1]  # Exclude Q[H] which is all zeros
    
    def optimal_value_function(self) -> List[np.ndarray]:
        """
        Compute the optimal value function V* using backward induction.
        
        Returns:
            List of optimal value function arrays V[h][s] for each time step h
        """
        # Initialize V[H] = 0
        V = [np.zeros(self.S) for _ in range(self.H+1)]
        
        # Backward induction
        for h in range(self.H-1, -1, -1):
            for s in range(self.S):
                # Find maximum Q-value over all actions
                max_q = -np.inf
                for a in range(self.A):
                    q = self.r[h][s,a]
                    for s_next in range(self.S):
                        q += self.P[h][s,a,s_next] * V[h+1][s_next]
                    if q > max_q:
                        max_q = q
                V[h][s] = max_q
        
        return V[:-1]  # Exclude V[H] which is all zeros
    
    def optimal_q_function(self) -> List[np.ndarray]:
        """
        Compute the optimal Q-function Q* using backward induction.
        
        Returns:
            List of optimal Q-function arrays Q[h][s,a] for each time step h
        """
        # Initialize Q[H] = r[H] since V[H+1] = 0
        Q = [np.zeros((self.S, self.A)) for _ in range(self.H+1)]
        
        # For the last step, Q is just the immediate reward
        for s in range(self.S):
            for a in range(self.A):
                Q[self.H-1][s,a] = self.r[self.H-1][s,a]
        
        # Backward induction
        for h in range(self.H-2, -1, -1):
            for s in range(self.S):
                for a in range(self.A):
                    # Immediate reward
                    Q[h][s,a] = self.r[h][s,a]
                    # Expected future value (taking optimal action at h+1)
                    for s_next in range(self.S):
                        Q[h][s,a] += self.P[h][s,a,s_next] * np.max(Q[h+1][s_next])
        
        return Q[:-1]  # Exclude Q[H] which is all zeros
    
    def get_optimal_policy(self, Q: List[np.ndarray] = None) -> List[np.ndarray]:
        """
        Get the optimal deterministic policy from the optimal Q-function.
        
        Args:
            Q: Optional Q-function (if None, computes optimal Q-function)
            
        Returns:
            List of deterministic policies where policy[h][s] gives optimal action at step h
        """
        if Q is None:
            Q = self.optimal_q_function()
        
        policy = []
        for h in range(self.H):
            policy_h = np.zeros(self.S, dtype=int)
            for s in range(self.S):
                policy_h[s] = np.argmax(Q[h][s])
            policy.append(policy_h)
        
        return policy
    
    def simulate_trajectory(self, policy: List[np.ndarray], initial_state: int) -> Tuple[List[int], List[int], float]:
        """
        Simulate a trajectory following the given policy from an initial state.
        
        Args:
            policy: List of deterministic policies where policy[h][s] gives action at step h
            initial_state: Initial state index
            
        Returns:
            Tuple of (state trajectory, action trajectory, total reward)
        """
        states = [initial_state]
        actions = []
        total_reward = 0.0
        
        for h in range(self.H):
            # Get action from policy
            a = policy[h][states[-1]]
            actions.append(a)
            
            # Add immediate reward
            total_reward += self.r[h][states[-1], a]
            
            # Transition to next state
            if h < self.H - 1:
                next_state = np.random.choice(self.S, p=self.P[h][states[-1], a])
                states.append(next_state)
        
        return states, actions, total_reward
