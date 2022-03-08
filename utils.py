import argparse

import gym
import numpy as np

from data_versions import DataVersions
from datastore import Datastore, CombinedDatastore
from feature_type import FeatureType
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


def get_datastore(data_version: DataVersions, feature_type: FeatureType = FeatureType.MFCC,
                  custom_split: float = None) -> Datastore:
    if data_version == DataVersions.IEMOCAP:
        from datastore_iemocap import IemocapDatastore

        return IemocapDatastore(feature_type, custom_split)

    if data_version == DataVersions.ESD:
        from datastore_esd import ESDDatastore
        return ESDDatastore(feature_type, custom_split)

    raise NotImplementedError(data_version)


def get_environment(data_version: DataVersions, datastore: Datastore, custom_split: float = None) -> gym.Env:
    if data_version == DataVersions.IEMOCAP:
        from environments import IemocapEnv, ESDEnv
        return IemocapEnv(data_version, datastore=datastore, custom_split=custom_split)

    if data_version == DataVersions.ESD:
        from environments import ESDEnv
        return ESDEnv(data_version, datastore=datastore, custom_split=custom_split)

    if data_version == DataVersions.COMBINED:
        from environments import CombinedEnv
        return CombinedEnv(data_version, datastore=datastore, custom_split=custom_split)

    raise NotImplementedError(data_version)


def combine_datastores(datastores: list) -> Datastore:
    ds1 = datastores[0]
    (x_train, y_train, _), _ = ds1.get_data()
    (x_target, y_target, _) = ds1.get_testing_data()

    for i in range(1, len(datastores)):
        ds2 = datastores[i]
        (x_train_2, y_train_2, _), _ = ds2.get_data()
        x_train = np.concatenate([x_train, x_train_2], axis=0)
        y_train = np.concatenate([y_train, y_train_2], axis=0)

        (x_target_2, y_target_2, _) = ds2.get_testing_data()
        x_target = np.concatenate([x_target, x_target_2], axis=0)
        y_target = np.concatenate([y_target, y_target_2], axis=0)

    # (x_train_1, y_train_1, _), _ = ds1.get_data()
    # (x_train_2, y_train_2, _), _ = ds2.get_data()
    #
    # x_train = np.concatenate([x_train_1, x_train_2], axis=0)
    # y_train = np.concatenate([y_train_1, y_train_2], axis=0)
    #
    # (x_target_1, y_target_1, _) = ds1.get_testing_data()
    # (x_target_2, y_target_2, _) = ds2.get_testing_data()
    # x_target = np.concatenate([x_target_1, x_target_2], axis=0)
    # y_target = np.concatenate([y_target_1, y_target_2], axis=0)

    return CombinedDatastore(x_train, y_train, x_target, y_target)


def store_results(filepath: str, args, experiment, time_str, test_loss, test_acc):
    content = f"Start Time:\t{time_str}\n" \
              f"Experiment:\t{experiment}\n"

    for k in args.__dict__.keys():
        argument_line = f"\t{k}: \t{args.__dict__[k]}\n"
        content += argument_line

    content += f"Test Loss: {test_loss}\n" \
               f"Test Accuracy: {test_acc}\n"

    write_to_file(filepath, content, overwrite=True)


def write_to_file(file: str, content: str, overwrite: bool = False):
    f = open(file, "w" if overwrite else "a")
    f.write(content)
    f.close()
