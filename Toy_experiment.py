import numpy as np
from tqdm import tqdm

class InvestmentPlanningMDP:
    def __init__(self, n_resource_levels=10, seed=42):
        np.random.seed(seed)
        self.n_resource_levels = n_resource_levels
        self.n_market_states = 3
        self.num_states = n_resource_levels * self.n_market_states
        self.num_actions = 3
        self.actions = [f'action{i}' for i in range(self.num_actions)]

        self.P = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.R = np.zeros((self.num_states, self.num_actions))

        self._build_transition_and_reward()

        self.initial_state_dist = np.zeros(self.num_states)
        for market in range(self.n_market_states):
            state_idx = self._get_state_idx(n_resource_levels // 2, market)
            self.initial_state_dist[state_idx] = 1.0 / self.n_market_states

        self.reset()

    def _get_state_idx(self, resource_level, market_state):
        return resource_level * self.n_market_states + market_state

    def _get_state_representation(self, state_idx):
        resource_level = state_idx // self.n_market_states
        market_state = state_idx % self.n_market_states
        return resource_level, market_state

    def _build_transition_and_reward(self):
        for state_idx in range(self.num_states):
            resource_level, market_state = self._get_state_representation(state_idx)

            for action in range(self.num_actions):
                if action == 0:
                    base_reward = 1.0
                    if market_state == 2:
                        base_reward *= 0.5
                elif action == 1:
                    if market_state == 0:
                        base_reward = 0.5
                    elif market_state == 1:
                        base_reward = 2.0
                    else:
                        base_reward = 1.5
                else:
                    if market_state == 0:
                        base_reward = -1.0
                    elif market_state == 1:
                        base_reward = 1.0
                    else:
                        base_reward = 3.0

                reward_scale = (resource_level + 1) / self.n_resource_levels
                self.R[state_idx, action] = base_reward * reward_scale

                if action == 0:
                    resource_change_probs = [0.1, 0.8, 0.1, 0.0, 0.0]
                elif action == 1:
                    resource_change_probs = [0.2, 0.2, 0.4, 0.2, 0.0]
                else:
                    resource_change_probs = [0.4, 0.1, 0.1, 0.2, 0.2]

                if market_state == 0:
                    market_trans_probs = [0.6, 0.3, 0.1]
                elif market_state == 1:
                    market_trans_probs = [0.3, 0.4, 0.3]
                else:
                    market_trans_probs = [0.1, 0.3, 0.6]

                for res_change_idx, res_change_prob in enumerate(resource_change_probs):
                    resource_change = res_change_idx - 1
                    next_resource = min(max(0, resource_level + resource_change), self.n_resource_levels - 1)

                    for next_market, market_prob in enumerate(market_trans_probs):
                        next_state_idx = self._get_state_idx(next_resource, next_market)
                        self.P[state_idx, action, next_state_idx] += res_change_prob * market_prob

    def reset(self):
        self.state = np.random.choice(self.num_states, p=self.initial_state_dist)
        return self.state

    def step(self, action_idx):
        transition_probs = self.P[self.state, action_idx]
        next_state = np.random.choice(self.num_states, p=transition_probs)
        reward = self.R[self.state, action_idx]
        self.state = next_state
        return next_state, reward, False

    def get_optimal_policy(self, gamma=0.9, theta=1e-6):
        V = np.zeros(self.num_states)

        while True:
            delta = 0
            for s in range(self.num_states):
                v = V[s]
                q_values = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    for s_prime in range(self.num_states):
                        q_values[a] += self.P[s, a, s_prime] * V[s_prime]
                    q_values[a] = self.R[s, a] + gamma * q_values[a]
                V[s] = np.max(q_values)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        Q = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            for a in range(self.num_actions):
                for s_prime in range(self.num_states):
                    Q[s, a] += self.P[s, a, s_prime] * V[s_prime]
                Q[s, a] = self.R[s, a] + gamma * Q[s, a]

        deterministic_policy = np.zeros(self.num_states, dtype=int)
        for s in range(self.num_states):
            deterministic_policy[s] = np.argmax(Q[s])

        optimal_policy = np.zeros((self.num_states, self.num_actions))
        for s in range(self.num_states):
            optimal_policy[s, deterministic_policy[s]] = 1.0

        return optimal_policy, V, Q

    def evaluate_policy(self, policy, gamma=0.9, n_episodes=100, max_steps=50):
        optimal_policy, V_star, Q_star = self.get_optimal_policy(gamma)
        total_returns = 0
        total_v_diff = 0

        for _ in range(n_episodes):
            state = self.reset()
            episode_return = 0
            discount = 1.0

            for step in range(max_steps):
                action = policy.select_action(state)
                next_state, reward, _ = self.step(action)
                episode_return += discount * reward
                discount *= gamma
                state = next_state

            total_returns += episode_return

            initial_state = np.random.choice(self.num_states, p=self.initial_state_dist)
            policy_probs = policy.get_policy_distribution(initial_state)
            policy_value = sum(policy_probs[a] * Q_star[initial_state, a] for a in range(self.num_actions))
            v_diff = V_star[initial_state] - policy_value
            total_v_diff += v_diff

        avg_return = total_returns / n_episodes
        avg_v_diff = total_v_diff / n_episodes
        return avg_return, avg_v_diff


class Policy:
    def __init__(self, num_states, num_actions):
        self.theta = np.zeros((num_states, num_actions))
        self.num_states = num_states
        self.num_actions = num_actions

    def get_action_prob(self, state):
        logits = self.theta[state]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def select_action(self, state):
        probs = self.get_action_prob(state)
        action = np.random.choice(self.num_actions, p=probs)
        return action

    def get_policy_distribution(self, state):
        return self.get_action_prob(state)


class RKHSPolicy:
    def __init__(self, num_states, num_actions, kernel='rbf', sigma=1.0, temperature=1.0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.kernel_type = kernel
        self.sigma = sigma
        self.temperature = temperature

        self.state_action_pairs = [(s, a) for s in range(num_states) for a in range(num_actions)]
        self.N = len(self.state_action_pairs)
        self.beta = np.zeros(self.N)
        self.K = self.compute_kernel_matrix()

    def compute_kernel_matrix(self):
        K = np.zeros((self.N, self.N))
        for i, (s1, a1) in enumerate(self.state_action_pairs):
            for j, (s2, a2) in enumerate(self.state_action_pairs):
                if self.kernel_type == 'rbf':
                    x1 = np.array([s1, a1])
                    x2 = np.array([s2, a2])
                    distance = np.linalg.norm(x1 - x2)
                    K[i, j] = np.exp(- (distance ** 2) / (2 * self.sigma ** 2))
                else:
                    raise NotImplementedError("only RBF")
        return K

    def f(self, s, a):
        idx = s * self.num_actions + a
        K_sa = self.K[idx, :]
        return np.dot(self.beta, K_sa)

    def get_action_prob(self, s):
        logits = np.array([self.f(s, a) for a in range(self.num_actions)]) * self.temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        return probs

    def select_action(self, s):
        probs = self.get_action_prob(s)
        action = np.random.choice(self.num_actions, p=probs)
        return action

    def get_policy_distribution(self, s):
        return self.get_action_prob(s)

    def get_kernel_feature_vector(self, s, a):
        idx = s * self.num_actions + a
        return self.K[idx, :]

    def update_beta(self, delta_beta):
        self.beta += delta_beta


def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    V = np.zeros(env.num_states)

    while True:
        delta = 0
        for s in range(env.num_states):
            v = V[s]
            new_v = 0
            policy_probs = policy.get_policy_distribution(s)

            for a in range(env.num_actions):
                if policy_probs[a] > 0:
                    expected_next_value = 0
                    for s_prime in range(env.num_states):
                        expected_next_value += env.P[s, a, s_prime] * V[s_prime]
                    new_v += policy_probs[a] * (env.R[s, a] + gamma * expected_next_value)

            V[s] = new_v
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    Q = np.zeros((env.num_states, env.num_actions))
    for s in range(env.num_states):
        for a in range(env.num_actions):
            Q[s, a] = env.R[s, a]
            for s_prime in range(env.num_states):
                Q[s, a] += gamma * env.P[s, a, s_prime] * V[s_prime]

    return V, Q


def compute_policy_distance(policy, optimal_policy, env):
    distance = 0
    for s in range(env.num_states):
        if hasattr(policy, 'get_policy_distribution'):
            policy_probs = policy.get_policy_distribution(s)
        else:
            policy_probs = policy[s]

        optimal_probs = optimal_policy[s]
        state_distance = 0.5 * np.sum(np.abs(policy_probs - optimal_probs))
        distance += state_distance / env.num_states

    return distance


def policy_gradient_with_policy_values(env, policy, num_iterations=1000, gamma=0.9, learning_rate=0.01):
    optimal_policy, V_star, Q_star = env.get_optimal_policy(gamma)
    
    metrics = {
        'returns': [],
        'v_diff': [],
        'policy_distance': []
    }

    avg_return, avg_v_diff = env.evaluate_policy(policy, gamma)
    metrics['returns'].append(avg_return)
    metrics['v_diff'].append(avg_v_diff)
    metrics['policy_distance'].append(compute_policy_distance(policy, optimal_policy, env))

    for iteration in tqdm(range(num_iterations), desc="PG with Policy Values"):
        V_pi, Q_pi = policy_evaluation(env, policy, gamma)
        batch_grads = np.zeros_like(policy.theta)

        for s in range(env.num_states):
            action_probs = policy.get_policy_distribution(s)
            for a in range(env.num_actions):
                Q_value = Q_pi[s, a]
                for a_prime in range(policy.num_actions):
                    if a_prime == a:
                        batch_grads[s, a_prime] += (1 - action_probs[a_prime]) * Q_value
                    else:
                        batch_grads[s, a_prime] += (-action_probs[a_prime]) * Q_value

        policy.theta += learning_rate * batch_grads

        if (iteration + 1) % 10 == 0:
            avg_return, avg_v_diff = env.evaluate_policy(policy, gamma)
            policy_dist = compute_policy_distance(policy, optimal_policy, env)
            metrics['returns'].append(avg_return)
            metrics['v_diff'].append(avg_v_diff)
            metrics['policy_distance'].append(policy_dist)

            if (iteration + 1) % 100 == 0:
                print(f"[PG-Policy] Iteration {iteration+1}/{num_iterations}, Avg Return: {avg_return:.2f}, "
                      f"V Diff: {avg_v_diff:.4f}, Policy Dist: {policy_dist:.4f}")

    return metrics


def policy_newton_with_policy_values(env, policy, num_iterations=1000, gamma=0.9, learning_rate=1.0, epsilon=1e-5):
    optimal_policy, V_star, Q_star = env.get_optimal_policy(gamma)
    
    metrics = {
        'returns': [],
        'v_diff': [],
        'policy_distance': []
    }

    avg_return, avg_v_diff = env.evaluate_policy(policy, gamma)
    metrics['returns'].append(avg_return)
    metrics['v_diff'].append(avg_v_diff)
    metrics['policy_distance'].append(compute_policy_distance(policy, optimal_policy, env))

    for iteration in tqdm(range(num_iterations), desc="PN with Policy Values"):
        V_pi, Q_pi = policy_evaluation(env, policy, gamma)
        batch_grads = np.zeros_like(policy.theta)
        batch_H = np.zeros((policy.num_states, policy.num_actions, policy.num_actions))

        for s in range(env.num_states):
            action_probs = policy.get_policy_distribution(s)

            for a in range(env.num_actions):
                Q_value = Q_pi[s, a]
                for a_prime in range(policy.num_actions):
                    if a_prime == a:
                        batch_grads[s, a_prime] += (1 - action_probs[a_prime]) * Q_value
                    else:
                        batch_grads[s, a_prime] += (-action_probs[a_prime]) * Q_value

            for a in range(policy.num_actions):
                for b in range(policy.num_actions):
                    Q_value_a = Q_pi[s, a]
                    if a == b:
                        batch_H[s, a, b] += action_probs[a] * (1 - action_probs[a]) * Q_value_a
                    else:
                        batch_H[s, a, b] += -action_probs[a] * action_probs[b] * Q_value_a

        for state_idx in range(policy.num_states):
            batch_H[state_idx] += epsilon * np.eye(policy.num_actions)

        try:
            delta_theta = np.zeros_like(policy.theta)
            for state_idx in range(policy.num_states):
                grad_vector = batch_grads[state_idx]
                H_matrix = batch_H[state_idx]
                delta = np.linalg.solve(H_matrix, grad_vector)
                delta_theta[state_idx] = delta
        except np.linalg.LinAlgError:
            print("Using GD")
            delta_theta = batch_grads

        policy.theta += learning_rate * delta_theta

        if (iteration + 1) % 10 == 0:
            avg_return, avg_v_diff = env.evaluate_policy(policy, gamma)
            policy_dist = compute_policy_distance(policy, optimal_policy, env)
            metrics['returns'].append(avg_return)
            metrics['v_diff'].append(avg_v_diff)
            metrics['policy_distance'].append(policy_dist)

            if (iteration + 1) % 100 == 0:
                print(f"[PN-Policy] Iteration {iteration+1}/{num_iterations}, Avg Return: {avg_return:.2f}, "
                      f"V Diff: {avg_v_diff:.4f}, Policy Dist: {policy_dist:.4f}")

    return metrics


def policy_gradient_rkhs_with_policy_values(env, policy, num_iterations=1000, gamma=0.9, learning_rate=0.01):
    optimal_policy, V_star, Q_star = env.get_optimal_policy(gamma)
    
    metrics = {
        'returns': [],
        'v_diff': [],
        'policy_distance': []
    }

    avg_return, avg_v_diff = env.evaluate_policy(policy, gamma)
    metrics['returns'].append(avg_return)
    metrics['v_diff'].append(avg_v_diff)
    metrics['policy_distance'].append(compute_policy_distance(policy, optimal_policy, env))

    for iteration in tqdm(range(num_iterations), desc="RKHS-PG with Policy Values"):
        V_pi, Q_pi = policy_evaluation(env, policy, gamma)
        batch_grads = np.zeros_like(policy.beta)

        for s in range(env.num_states):
            for a in range(env.num_actions):
                Q_value = Q_pi[s, a]
                K_sa = policy.get_kernel_feature_vector(s, a)
                action_probs = policy.get_policy_distribution(s)
                K_features = np.array([policy.get_kernel_feature_vector(s, a_prime) for a_prime in range(policy.num_actions)])
                K_expected = np.dot(action_probs, K_features)
                batch_grads += policy.temperature * (K_sa - K_expected) * Q_value * action_probs[a]

        policy.beta += learning_rate * batch_grads

        if (iteration + 1) % 10 == 0:
            avg_return, avg_v_diff = env.evaluate_policy(policy, gamma)
            policy_dist = compute_policy_distance(policy, optimal_policy, env)
            metrics['returns'].append(avg_return)
            metrics['v_diff'].append(avg_v_diff)
            metrics['policy_distance'].append(policy_dist)

            if (iteration + 1) % 100 == 0:
                print(f"[RKHS-PG-Policy] Iteration {iteration+1}/{num_iterations}, Avg Return: {avg_return:.2f}, "
                      f"V Diff: {avg_v_diff:.4f}, Policy Dist: {policy_dist:.4f}")

    return metrics


def policy_newton_rkhs_with_policy_values(env, policy, num_iterations=1000, gamma=0.9, learning_rate=0.1, epsilon=1e-5):
    optimal_policy, V_star, Q_star = env.get_optimal_policy(gamma)
    
    metrics = {
        'returns': [],
        'v_diff': [],
        'policy_distance': []
    }

    avg_return, avg_v_diff = env.evaluate_policy(policy, gamma)
    metrics['returns'].append(avg_return)
    metrics['v_diff'].append(avg_v_diff)
    metrics['policy_distance'].append(compute_policy_distance(policy, optimal_policy, env))

    for iteration in tqdm(range(num_iterations), desc="RKHS-PN with Policy Values"):
        V_pi, Q_pi = policy_evaluation(env, policy, gamma)
        batch_grads = np.zeros_like(policy.beta)
        H_k = np.zeros((policy.N, policy.N))

        for s in range(env.num_states):
            for a in range(env.num_actions):
                Q_value = Q_pi[s, a]
                action_probs = policy.get_policy_distribution(s)
                K_sa = policy.get_kernel_feature_vector(s, a)
                K_features = np.array([policy.get_kernel_feature_vector(s, a_prime) for a_prime in range(policy.num_actions)])
                K_expected = np.dot(action_probs, K_features)

                grad = policy.temperature * (K_sa - K_expected) * Q_value * action_probs[a]
                batch_grads += grad

                grad_outer = np.outer(grad, grad)
                cov = np.zeros((policy.N, policy.N))
                for a_prime in range(policy.num_actions):
                    K_a_prime = policy.get_kernel_feature_vector(s, a_prime)
                    cov += action_probs[a_prime] * np.outer(K_a_prime, K_a_prime)
                cov -= np.outer(K_expected, K_expected)
                H_k += grad_outer + policy.temperature * Q_value * action_probs[a] * cov

        H_k += epsilon * np.eye(policy.N)

        try:
            delta_beta = np.linalg.solve(H_k, batch_grads)
        except np.linalg.LinAlgError:
            print("Hessian matrix is not invertible, using gradient descent")
            delta_beta = batch_grads

        policy.update_beta(learning_rate * delta_beta)

        if (iteration + 1) % 10 == 0:
            avg_return, avg_v_diff = env.evaluate_policy(policy, gamma)
            policy_dist = compute_policy_distance(policy, optimal_policy, env)
            metrics['returns'].append(avg_return)
            metrics['v_diff'].append(avg_v_diff)
            metrics['policy_distance'].append(policy_dist)

            if (iteration + 1) % 100 == 0:
                print(f"[RKHS-PN-Policy] Iteration {iteration+1}/{num_iterations}, Avg Return: {avg_return:.2f}, "
                      f"V Diff: {avg_v_diff:.4f}, Policy Dist: {policy_dist:.4f}")

    return metrics


def run_all_algorithms(num_iterations=200, gamma=0.9):
    """Run all four algorithms and return their metrics"""
    
    env = InvestmentPlanningMDP(n_resource_levels=2, seed=42)
    
    method_names = [
        "Policy Gradient", 
        "Policy Newton", 
        "Policy Gradient in RKHS", 
        "Policy Newton in RKHS"
    ]
    
    print("Running all algorithms...")
    
    # Initialize policies
    policy_pg = Policy(env.num_states, env.num_actions)
    policy_pn = Policy(env.num_states, env.num_actions)
    policy_rkhs_pg = RKHSPolicy(env.num_states, env.num_actions, kernel='rbf', sigma=0.01, temperature=1.0)
    policy_rkhs_pn = RKHSPolicy(env.num_states, env.num_actions, kernel='rbf', sigma=0.01, temperature=1.0)
    
    # Run algorithms
    print(f"Running {method_names[0]}...")
    metrics_pg = policy_gradient_with_policy_values(
        env, policy_pg, num_iterations=num_iterations, gamma=gamma, learning_rate=0.01)
    
    print(f"Running {method_names[1]}...")
    metrics_pn = policy_newton_with_policy_values(
        env, policy_pn, num_iterations=num_iterations, gamma=gamma, learning_rate=1.0, epsilon=1e-5)
    
    print(f"Running {method_names[2]}...")
    metrics_rkhs_pg = policy_gradient_rkhs_with_policy_values(
        env, policy_rkhs_pg, num_iterations=num_iterations, gamma=gamma, learning_rate=1.0)
    
    print(f"Running {method_names[3]}...")
    metrics_rkhs_pn = policy_newton_rkhs_with_policy_values(
        env, policy_rkhs_pn, num_iterations=num_iterations, gamma=gamma, learning_rate=100.0, epsilon=1e-5)
    
    # Collect results
    all_metrics = [metrics_pg, metrics_pn, metrics_rkhs_pg, metrics_rkhs_pn]
    
    # Print final results
    print("\n=== Final Results ===")
    for i, (name, metrics) in enumerate(zip(method_names, all_metrics)):
        final_return = metrics['returns'][-1]
        final_v_diff = metrics['v_diff'][-1]
        final_policy_dist = metrics['policy_distance'][-1]
        print(f"{name}:")
        print(f"  Final Return: {final_return:.4f}")
        print(f"  Final V Diff: {final_v_diff:.6f}")
        print(f"  Final Policy Distance: {final_policy_dist:.6f}")
        print()
    
    return all_metrics, method_names


if __name__ == "__main__":
    # Run the experiment
    metrics_results, method_names = run_all_algorithms(num_iterations=200, gamma=0.9)
    
    print("Experiment completed successfully!")