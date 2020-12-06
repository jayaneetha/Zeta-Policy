import numpy as np

from rl.policy import Policy


class ZetaPolicy(Policy):
    """Implement the Zeta Q Policy
    """

    def __init__(self, zeta_start=1.0, zeta_end=.1, zeta_nb_steps=1000000, eps=.1, tau=1., clip=(-500., 500.)):
        super(ZetaPolicy, self).__init__()
        self.zeta_start = zeta_start
        self.zeta_end = zeta_end
        self.zeta_nb_steps = zeta_nb_steps
        self.eps = eps
        self.tau = tau
        self.clip = clip

    def get_zeta(self):
        a = - ((self.zeta_start - self.zeta_end) / float(self.zeta_nb_steps))
        b = self.zeta_start
        return max(self.zeta_end, a * self.agent.step + b)

    def select_action(self, q_values):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1

        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        def get_action_from_boltzmann(q, tau, clip):
            exp_values = np.exp(np.clip(q / tau, clip[0], clip[1]))
            probs = exp_values / np.sum(exp_values)
            a = np.random.choice(range(nb_actions), p=probs)
            return a

        def get_action_from_maxboltzmann(q, tau, clip, eps):
            if np.random.uniform() < eps:
                a = get_action_from_boltzmann(q, tau, clip)
            else:
                a = np.argmax(q_values)
            return a

        zeta = self.get_zeta()

        if zeta > self.zeta_end:
            action = get_action_from_maxboltzmann(q_values, self.tau, self.clip, self.eps)
        else:
            if np.random.uniform() < self.eps:
                action = np.random.randint(0, nb_actions)
            else:
                action = np.argmax(q_values)

        return action

    def get_config(self):
        """Return configurations of CustomPolicy

        # Returns
            Dict of config
        """
        config = super(ZetaPolicy, self).get_config()
        config['zeta_start'] = self.zeta_start
        config['zeta_end'] = self.zeta_end
        config['zeta_nb_steps'] = self.zeta_nb_steps
        config['eps'] = self.eps
        config['tau'] = self.tau
        config['clip'] = self.clip
        return config
