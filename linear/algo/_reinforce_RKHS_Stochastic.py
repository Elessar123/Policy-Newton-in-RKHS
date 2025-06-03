import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import softmax
from scipy.spatial.distance import cdist
from ._agent import LinearRLAgent


class RKHSLinearPolySGD(LinearRLAgent):
    def __init__(
        self,
        envs,
        lr=0.01,
        normalize_returns=False,
        poly_degree=1,
        set_bias=False,
        kernel='rbf',
        kernel_params=None,
        temperature=1.0
    ):
        super().__init__(envs)
        
        self.lr = lr
        self.temperature = temperature  
        self.normalize_returns = normalize_returns
        
        self.featurize = PolynomialFeatures(degree=poly_degree, include_bias=set_bias, order='F')
        sample_observation = self.observation_space.sample().reshape(1, -1)
        self.featurize.fit(sample_observation)
        
        self.num_features = self.featurize.n_output_features_
        self.num_actions = self.action_space.n
        
        self.kernel = kernel
        self.kernel_params = kernel_params if kernel_params is not None else {}
        if self.kernel == 'rbf':
            self.gamma = self.kernel_params.get('gamma', 1.0)
        elif self.kernel == 'poly':
            self.degree_kernel = self.kernel_params.get('degree', 3)
            self.coef0 = self.kernel_params.get('coef0', 1)
        else:
            raise NotImplementedError(f"Kernel '{self.kernel}' not implemented.")
        
        self.centers = []  
        self.betas = []     

    @property
    def params(self):
        return np.array(self.betas, dtype=np.float32)

    @params.setter
    def params(self, new_params):
        self.betas = list(new_params)

    def get_state(self, observations):

        return self.featurize.transform(observations)

    def one_hot_encode_actions(self, actions):

        num_actions = self.num_actions
        one_hot = np.zeros((actions.shape[0], num_actions), dtype=np.float32)
        one_hot[np.arange(actions.shape[0]), actions.flatten()] = 1.0
        return one_hot

    def _compute_kernel(self, s_a):

        if not self.centers:
            return np.zeros((s_a.shape[0], 0), dtype=np.float32)
        
        centers_array = np.array(self.centers)  # Shape: (N, feature_dim * num_actions)
        
        if self.kernel == 'rbf':
            dists = cdist(s_a, centers_array, 'sqeuclidean')
            K = np.exp(-self.gamma * dists)
        elif self.kernel == 'poly':
            K = (self.gamma * np.dot(s_a, centers_array.T) + self.coef0) ** self.degree_kernel
        else:
            raise NotImplementedError(f"Kernel '{self.kernel}' not implemented.")
        
        return K  # Shape: (num_samples, N)

    def get_action(self, state):
        state_features = self.get_state(state)  # Shape: (batch_size, num_features)
        
        batch_size = state_features.shape[0]
        
        actions = np.arange(self.num_actions)  # [0,1,...,num_actions-1]
        
        actions_one_hot = self.one_hot_encode_actions(actions.reshape(-1, 1))  # Shape: (num_actions, num_actions)
        
        state_expanded = np.repeat(state_features, self.num_actions, axis=0)  # Shape: (batch_size * num_actions, num_features)
        
        actions_expanded = np.tile(actions_one_hot, (batch_size, 1))  # Shape: (batch_size * num_actions, num_actions)
        
        s_a = self._mult1(actions_expanded, state_expanded)  # Shape: (batch_size * num_actions, num_features * num_actions)
        
        K = self._compute_kernel(s_a)  # Shape: (batch_size * num_actions, N)
        
        if len(self.betas) > 0:
            f_sa = np.dot(K, self.params)  # Shape: (batch_size * num_actions,)
        else:
            f_sa = np.zeros(batch_size * self.num_actions, dtype=np.float32)
        
        logits = f_sa.reshape(batch_size, self.num_actions)
        
        action_prob = softmax(logits, axis=1)  # Shape: (batch_size, num_actions)

        cdf = np.cumsum(action_prob, axis=1)  # Shape: (batch_size, num_actions)
        rvs = np.random.rand(batch_size, 1)  # Shape: (batch_size, 1)
        action = np.argmax(rvs < cdf, axis=1)  # Shape: (batch_size,)
        
        return action, action_prob

    def _get_action_probabilities(self, state_features):

        batch_size = state_features.shape[0]
        logits = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        
        for a in range(self.num_actions):
            a_one_hot = self.one_hot_encode_actions(np.array([a]))  # Shape: (1, num_actions)
            
            s_a = self._mult1(a_one_hot, state_features)  # Shape: (batch_size, num_features * num_actions)
            
            K = self._compute_kernel(s_a)  # Shape: (batch_size, N)
            
            if len(self.betas) > 0:
                f_sa = np.dot(K, self.params)  # Shape: (batch_size,)
            else:
                f_sa = np.zeros(batch_size, dtype=np.float32)
            
            logits[:, a] = f_sa
        
        action_prob = softmax(logits, axis=1)  # Shape: (batch_size, num_actions)
        return action_prob

    def learn(self, traj_info, dones, lr=None):

        LEARNING_RATE = self.lr if lr is None else lr
        TEMPERATURE = self.temperature  
        
        S = traj_info['states']       # Shape: (T, batch_size, state_dim)
        A = traj_info['actions']      # Shape: (T, batch_size)
        P = traj_info['action_probs'] # Shape: (T, batch_size, num_actions)
        R = traj_info['rewards']      # Shape: (T, batch_size)
        
        Y = self.discount_cumsum(R, dones, gamma=self.discount_factor, normalize=self.normalize_returns)  # Shape: (T, batch_size)
        
        T_steps, batch_size = Y.shape
        batch_size_predefine = 1
        T_steps_predefine = 5
        random_T_steps = np.random.choice(np.arange(T_steps), size=T_steps_predefine, replace=False)
        # b = 0
        b = np.random.choice(np.arange(batch_size), size=1, replace=False)
        for t_step in range(len(random_T_steps)):
            t = random_T_steps[t_step]
            y = Y[t, b]  
            
            s = S[t, b]  
            a = A[t, b]   
            
            a_one_hot = self.one_hot_encode_actions(np.array([a]))  # Shape: (1, num_actions)
            
            s_encoded = self.get_state(s.reshape(1, -1))  # Shape: (1, num_features)
            s_a = self._mult1(a_one_hot, s_encoded)  # Shape: (1, num_features * num_actions)
            
            action_prob = self._get_action_probabilities(s_encoded).flatten()  # Shape: (num_actions,)
            # print(f"The action prob is {action_prob}")
            pi_a = action_prob[a]  # π(a|s)
            
            for a_prime in range(self.num_actions):
                if a_prime == a:
                    coefficient = TEMPERATURE * (1 - pi_a)
                else:
                    coefficient = TEMPERATURE * (-action_prob[a_prime])
                
                a_prime_one_hot = self.one_hot_encode_actions(np.array([a_prime]))  # Shape: (1, num_actions)
                s_a_prime = self._mult1(a_prime_one_hot, s_encoded)  # Shape: (1, num_features * num_actions)
                
                self.centers.append(s_a_prime.flatten())  # Store the flattened (s, a') features
                
                beta_update = LEARNING_RATE * y * coefficient
                # print(f"beta updated are {beta_update}")
                self.betas.append(beta_update)
                
                # print(f"Added new center. Total centers: {len(self.centers)}, betas length: {len(self.betas)}")
        
        return None, None, None
