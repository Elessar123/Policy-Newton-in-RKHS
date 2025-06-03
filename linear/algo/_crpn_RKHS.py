import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import softmax
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from ._agent import LinearRLAgent


class RKHSLinearPolyCRPN(LinearRLAgent):
    def __init__(
        self,
        envs,
        alpha=1e4,
        normalize_returns=False,
        poly_degree=1,
        set_bias=False,
        kernel='rbf',
        kernel_params=None,
        sigma=1.0,
        addition_per_iteration=5,
        predefined_T_steps=None,
        discount_factor=0.99,
        seed=1234,
        learning_rate=0.01,
        temperature=100.0
    ):
        super().__init__(envs)
        
        self.alpha = alpha
        self.normalize_returns = normalize_returns
        self.sigma = sigma
        self.addition_per_iteration = addition_per_iteration
        self.predefined_T_steps = predefined_T_steps
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.lr = learning_rate
        
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
        self.alpha_params = []
        
        np.random.seed(seed)

    @property
    def params(self):
        return np.array(self.alpha_params, dtype=np.float32)
    
    @params.setter
    def params(self, new_params):
        self.alpha_params = list(new_params)

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
        
        centers_array = np.array(self.centers)
        
        if self.kernel == 'rbf':
            dists = cdist(s_a, centers_array, 'sqeuclidean')
            K = np.exp(-self.gamma * dists)
        elif self.kernel == 'poly':
            K = (self.gamma * np.dot(s_a, centers_array.T) + self.coef0) ** self.degree_kernel
        else:
            raise NotImplementedError(f"Kernel '{self.kernel}' not implemented.")
        
        return K

    def _compute_kernel_each(self, s_a_1, s_a_2):
        if self.kernel == 'rbf':
            dists = cdist(s_a_1, s_a_2, 'sqeuclidean')
            K = np.exp(-self.gamma * dists)
        else:
            raise NotImplementedError(f"Kernel '{self.kernel}' not implemented.")
        return K

    def get_action(self, state):
        state_features = self.get_state(state)
        batch_size = state_features.shape[0]
        
        actions = np.arange(self.num_actions)
        actions_one_hot = self.one_hot_encode_actions(actions.reshape(-1, 1))
        state_expanded = np.repeat(state_features, self.num_actions, axis=0)
        actions_expanded = np.tile(actions_one_hot, (batch_size, 1))
        
        s_a = self._mult1(actions_expanded, state_expanded)
        K = self._compute_kernel(s_a)
        
        if len(self.alpha_params) > 0:
            f_sa = np.dot(K, self.params)
        else:
            f_sa = np.zeros(batch_size * self.num_actions, dtype=np.float32)
        
        logits = f_sa.reshape(batch_size, self.num_actions)
        action_prob = softmax(logits, axis=1)
        
        cdf = np.cumsum(action_prob, axis=1)
        rvs = np.random.rand(batch_size, 1)
        action = np.argmax(rvs < cdf, axis=1)
        
        return action, action_prob

    def _get_action_probabilities(self, state_features):
        batch_size = state_features.shape[0]
        logits = np.zeros((batch_size, self.num_actions), dtype=np.float32)
        
        for a in range(self.num_actions):
            a_one_hot = self.one_hot_encode_actions(np.array([a]))
            s_a = self._mult1(a_one_hot, state_features)
            K = self._compute_kernel(s_a)
            
            if len(self.alpha_params) > 0:
                f_sa = np.dot(K, self.params)
            else:
                f_sa = np.zeros(batch_size, dtype=np.float32)
            
            logits[:, a] = f_sa
        
        return softmax(logits, axis=1)

    def _fg(self, alpha, H, g, alpha_reg):
        norm_alpha = np.linalg.norm(alpha)
        if norm_alpha < 1e-8:
            norm_alpha = 1e-8
        
        Hv = H @ alpha
        s = np.dot(g, alpha) + 0.5 * np.dot(Hv, alpha) + (alpha_reg / 6) * norm_alpha ** 3
        j = g + Hv + (alpha_reg / 2) * norm_alpha * alpha
        return s, j

    def _hess(self, alpha, H, g, alpha_reg):
        norm_alpha = np.linalg.norm(alpha)
        if norm_alpha < 1e-8:
            norm_alpha = 1e-8
        outer_alpha = np.outer(alpha, alpha)
        hessian = H + (alpha_reg / 2) * (outer_alpha / norm_alpha + norm_alpha * np.eye(len(alpha)))
        return hessian

    def learn_second_order(self, traj_info, dones, alp=None):

        GAMMA = self.discount_factor
        ALPHA_REG = self.alpha if alp is None else alp
        
        S = traj_info['states']
        A = traj_info['actions']
        P = traj_info['action_probs']
        R = traj_info['rewards']
        
        T_steps, batch_size, _ = S.shape
        
        predefined_update_T = 50
        update_batch = 50
        update_T = min(predefined_update_T, T_steps)
        
        random_T_steps = np.random.choice(np.arange(T_steps), size=update_T, replace=False)
        random_batch = np.random.choice(np.arange(batch_size), size=update_batch, replace=False)
        random_center = np.random.choice(np.arange(batch_size), size=1, replace=False)
        
        S = S[:,random_batch,:]
        A = A[:,random_batch]
        P = P[:,random_batch,:]
        R = R[:,random_batch]
        dones = dones[:,random_batch]
        
        Center_all_S = S[random_T_steps,random_center].reshape(-1, S.shape[-1])
        Center_all_A = A[random_T_steps,random_center].reshape(-1)
        
        Y = self.discount_cumsum(R, dones, gamma=GAMMA, normalize=self.normalize_returns)
        Y = Y[random_T_steps]
        
        all_Y = Y.reshape(-1)
        all_S = S[random_T_steps].reshape(-1, S.shape[-1])
        all_A = A[random_T_steps].reshape(-1)
        all_P = P[random_T_steps].reshape(-1, P.shape[-1])
        
        if not self.centers:
            actual_predefined_T_steps = min(self.predefined_T_steps, len(all_S))
            if actual_predefined_T_steps == 0:
                return False, None, None, None
                
            centers_indices = np.linspace(0, len(all_S) - 1, num=actual_predefined_T_steps, dtype=int)
            centers_S = all_S[centers_indices]
            centers_A = all_A[centers_indices]
            
            centers_one_hot = self.one_hot_encode_actions(centers_A.reshape(-1, 1))
            centers_features = self.get_state(centers_S)
            centers_s_a = self._mult1(centers_one_hot, centers_features)
            
            self.centers = list(centers_s_a)
            self.alpha_params = [0.0 for _ in self.centers]
            
            G = np.zeros(len(self.centers), dtype=np.float32)
            H = np.eye(len(self.centers), dtype=np.float32)
            return False, G, H, None

        all_state_features = self.get_state(all_S)
        centers_one_hot = self.one_hot_encode_actions(Center_all_A.reshape(-1, 1))
        centers_features = self.get_state(Center_all_S)
        new_centers_s_a = self._mult1(centers_one_hot, centers_features)
        
        num_total = update_T * update_batch
        K_data = np.zeros((num_total, self.num_actions, update_T), dtype=np.float32)
        
        for a_prime in range(self.num_actions):
            a_prime_one_hot = self.one_hot_encode_actions(np.array([a_prime]))
            a_prime_expanded = np.repeat(a_prime_one_hot, num_total, axis=0)
            s_a_prime = self._mult1(a_prime_expanded, all_state_features)
            K_a_prime = self._compute_kernel_each(s_a_prime, new_centers_s_a)
            K_data[:, a_prime, :] = K_a_prime
        
        E_K = np.einsum('na,nak->nk', all_P, K_data)
        
        K_sa_i = np.zeros((update_T, update_batch, update_T), dtype=np.float32)
        for t in range(update_T):
            for b in range(update_batch):
                a_t = A[t, b]
                K_sa_i[t, b, :] = K_data[t * update_batch + b, a_t, :]
        
        delta_K = K_sa_i - E_K.reshape(update_T, update_batch, update_T)
        weighted_delta_K = delta_K * Y[:, :, np.newaxis]
        G = np.sum(weighted_delta_K, axis=(0, 1))
        
        K_data_expanded = K_data.reshape(update_T*update_batch, self.num_actions, update_T, 1)
        K_data_j = K_data.reshape(update_T*update_batch, self.num_actions, 1, update_T)
        K_product = K_data_expanded * K_data_j
        E_KK = np.sum(all_P[:, :, np.newaxis, np.newaxis] * K_product, axis=1)
        E_K_flat = E_K.reshape(update_T*update_batch, update_T, 1)
        E_KE_K = E_K_flat * E_K_flat.transpose(0, 2, 1)
        Cov = E_KK - E_KE_K
        
        b = np.sum(all_Y[:, np.newaxis] * delta_K.reshape(-1, update_T), axis=0)
        c = np.sum(delta_K, axis=(0,1))
        
        H = np.outer(b, c) - np.sum(all_Y[:, np.newaxis, np.newaxis] * Cov, axis=0)
        H += 1e-4 * np.eye(H.shape[0])
        
        v0 = np.zeros(G.shape[0], dtype=np.float32)
        result = minimize(
            fun=lambda alpha: self._fg(alpha, H, G, ALPHA_REG),
            x0=v0,
            method='Newton-CG',
            jac=True,
            hess=lambda alpha: self._hess(alpha, H, G, ALPHA_REG),
            tol=1e-3,
            options={'maxiter': 500}
        )
        
        if not result.success:
            return False, G, H, result
        
        alpha_opt = result.x.astype(np.float32)
        if np.max(np.abs(alpha_opt))>3:
            return False, G, H, result
        # alpha_opt = np.clip(alpha_opt,-1,1)
        print(f"Optimized alpha: {alpha_opt}")
        centers_new_S = all_S
        centers_new_A = all_A
        centers_new_one_hot = self.one_hot_encode_actions(centers_new_A.reshape(-1, 1))
        centers_new_features = self.get_state(centers_new_S)
        centers_new_s_a = self._mult1(centers_new_one_hot, centers_new_features)
        
        self.add_new_kernel_centers(centers_new_s_a, alpha_opt)
        
        return True, G, H, result

    def learn_first_order(self, traj_info, dones, lr=None):
        LEARNING_RATE = self.lr if lr is None else lr
        TEMPERATURE = self.temperature
        
        S = traj_info['states']
        A = traj_info['actions']
        P = traj_info['action_probs']
        R = traj_info['rewards']
        
        Y = self.discount_cumsum(R, dones, gamma=self.discount_factor, normalize=self.normalize_returns)
        
        T_steps, batch_size = Y.shape
        batch_size_predefine = 50
        T_steps_predefine = 3
        random_T_steps = np.random.choice(np.arange(T_steps), size=T_steps_predefine, replace=False)
        random_b = np.random.choice(np.arange(batch_size), size=batch_size_predefine, replace=False)
        
        for b_batch in range(len(random_b)):
            for t_step in range(len(random_T_steps)):
                t = random_T_steps[t_step]
                b = random_b[b_batch]
                y = Y[t, b]
                
                s = S[t, b]
                a = A[t, b]
                
                a_one_hot = self.one_hot_encode_actions(np.array([a]))
                s_encoded = self.get_state(s.reshape(1, -1))
                s_a = self._mult1(a_one_hot, s_encoded)
                
                action_prob = self._get_action_probabilities(s_encoded).flatten()
                pi_a = action_prob[a]
                
                for a_prime in range(self.num_actions):
                    if a_prime == a:
                        coefficient = TEMPERATURE * (1 - pi_a)
                    else:
                        coefficient = TEMPERATURE * (-action_prob[a_prime])
                    
                    a_prime_one_hot = self.one_hot_encode_actions(np.array([a_prime]))
                    s_a_prime = self._mult1(a_prime_one_hot, s_encoded)
                    
                    # self.centers.append(s_a_prime.flatten())
                    beta_update = LEARNING_RATE * y * coefficient
                    self.add_new_kernel_centers_single(s_a_prime.flatten(), beta_update)
        
        return None, None, None

    def learn(self, traj_info, dones, lr=None, alp=None, ):
        success, G, H, result = self.learn_second_order(traj_info, dones, alp)
        # np.max(result.x.astype(np.float32))<1e-3
        if not success:
            print("Second-order optimization failed, switching to first-order method...")
            return self.learn_first_order(traj_info, dones, lr)
        
        if success:
            if np.max(result.x.astype(np.float32))<1e-3:
                print("Second-order optimization small, switching to first-order method...")
                return self.learn_first_order(traj_info, dones, lr)
        return G, H, result

    def add_new_kernel_centers(self, new_centers_s_a, new_alpha_opts):
        for s_a_feature, alpha in zip(new_centers_s_a, new_alpha_opts):
            if np.abs(alpha) < 1e-3:
                continue
            self.centers.append(s_a_feature)
            self.alpha_params.append(alpha)

    def add_new_kernel_centers_single(self, s_a_feature, alpha):
        if np.abs(alpha) > 1e-3:
            self.centers.append(s_a_feature)
            self.alpha_params.append(alpha)
    
    @staticmethod
    def _mult1(A, B):

        return np.einsum('ij,ik->ijk', A, B).reshape(A.shape[0], -1)
    
    @staticmethod
    def _mult2(A, B):

        return np.einsum('ij,kl->ikjl', A, B).reshape(A.shape[0]*A.shape[1], A.shape[0]*A.shape[1])
    
    @staticmethod
    def _diagonalize(A):

        A_ = np.stack([A] * A.shape[-1], axis=-1)
        return A_ * np.eye(A.shape[-1])
    
    @staticmethod
    def discount_cumsum(rewards, dones, gamma, normalize=True):

        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = np.zeros_like(rewards[0])
        t = -1
        for r in rewards[::-1]:
            cumulative_reward = r + cumulative_reward * gamma  # Discount factor
            discounted_rewards[t, :] = cumulative_reward.copy()
            t -= 1
        if normalize:
            for i in range(rewards.shape[1]):
                m = np.argmax(dones[:, i]) - 1
                if m >= 0 and m < discounted_rewards.shape[0]:
                    mean = discounted_rewards[:m+1, i].mean()
                    std = discounted_rewards[:m+1, i].std()
                    discounted_rewards[:m+1, i] = (discounted_rewards[:m+1, i] - mean) / (std + 1e-9)
                elif m == -1:
                    discounted_rewards[:, i] = 0.0  
        return discounted_rewards * ~dones