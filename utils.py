import argparse

from data_versions import DataVersions
from rl.policy import MaxBoltzmannQPolicy, Policy, LinearAnnealedPolicy, EpsGreedyQPolicy, SoftmaxPolicy, GreedyQPolicy, \
    BoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl_custom_policy import ZetaPolicy


def parse_policy(args) -> Policy:
    pol: Policy = EpsGreedyQPolicy()
    if args.policy == 'LinearAnnealedPolicy':
        pol = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=0.05,
                                   nb_steps=args.zeta_nb_steps)
    if args.policy == 'SoftmaxPolicy':
        pol = SoftmaxPolicy()
    if args.policy == 'EpsGreedyQPolicy':
        pol = EpsGreedyQPolicy()
    if args.policy == 'GreedyQPolicy':
        pol = GreedyQPolicy()
    if args.policy == 'BoltzmannQPolicy':
        pol = BoltzmannQPolicy()
    if args.policy == 'MaxBoltzmannQPolicy':
        pol = MaxBoltzmannQPolicy()
    if args.policy == 'BoltzmannGumbelQPolicy':
        pol = BoltzmannGumbelQPolicy()
    if args.policy == 'ZetaPolicy':
        pol = ZetaPolicy(zeta_nb_steps=args.zeta_nb_steps, eps=args.eps)

    return pol


def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2dataset(v) -> DataVersions:
    ds = v.lower()
    if ds == 'iemocap':
        return DataVersions.IEMOCAP
    if ds == 'savee':
        return DataVersions.SAVEE
    if ds == 'improv':
        return DataVersions.IMPROV
    if ds == 'esd':
        return DataVersions.ESD
