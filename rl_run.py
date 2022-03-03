import argparse
import os

import gym
import tensorflow as tf
from tensorflow.keras import Input

import models
from constants import NUM_MFCC, NO_features, WINDOW_LENGTH
from data_versions import DataVersions
from datastore import Datastore
from environments import IemocapEnv, SaveeEnv, ImprovEnv, ESDEnv
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger
from rl.memory import SequentialMemory
from rl.policy import MaxBoltzmannQPolicy, Policy, LinearAnnealedPolicy, EpsGreedyQPolicy, SoftmaxPolicy, GreedyQPolicy, \
    BoltzmannQPolicy, BoltzmannGumbelQPolicy
from rl_custom_policy import ZetaPolicy


# from keras.optimizers import Adam


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


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='iemocap-rl-v3.1')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--policy', type=str, default='EpsGreedyQPolicy')
    parser.add_argument('--data-version',
                        choices=[DataVersions.IEMOCAP, DataVersions.SAVEE, DataVersions.IMPROV, DataVersions.ESD],
                        type=str2dataset, default=DataVersions.IEMOCAP)
    parser.add_argument('--disable-wandb', type=str2bool, default=False)
    parser.add_argument('--zeta-nb-steps', type=int, default=100000)
    parser.add_argument('--nb-steps', type=int, default=500000)
    parser.add_argument('--max-train-steps', type=int, default=440000)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--pre-train', type=str2bool, default=False)
    parser.add_argument('--pre-train-dataset',
                        choices=[DataVersions.IEMOCAP, DataVersions.IMPROV, DataVersions.SAVEE, DataVersions.ESD],
                        type=str2dataset,
                        default=DataVersions.IEMOCAP)
    parser.add_argument('--warmup-steps', type=int, default=50000)
    parser.add_argument('--pretrain-epochs', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(tf.__version__)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    # config = tf.ConfigProto(gpu_options=gpu_options)
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # tf.compat.v1.keras.backend.set_session(sess)

    policy = parse_policy(args)
    data_version = args.data_version

    env: gym.Env = None

    if data_version == DataVersions.IEMOCAP:
        env = IemocapEnv(data_version)

    if data_version == DataVersions.SAVEE:
        env = SaveeEnv(data_version)

    if data_version == DataVersions.IMPROV:
        env = ImprovEnv(data_version)

    if data_version == DataVersions.ESD:
        env = ESDEnv(data_version)

    for k in args.__dict__.keys():
        print("\t{} :\t{}".format(k, args.__dict__[k]))
        env.__setattr__("_" + k, args.__dict__[k])

    experiment_name = "P-{}-S-{}-e-{}-pt-{}".format(args.policy, args.zeta_nb_steps, args.eps, args.pre_train)
    if args.pre_train:
        experiment_name = "P-{}-S-{}-e-{}-pt-{}-pt-w-{}".format(args.policy, args.zeta_nb_steps, args.eps,
                                                                args.pre_train,
                                                                args.pre_train_dataset.name)
    env.__setattr__("_experiment", experiment_name)

    nb_actions = env.action_space.n

    input_layer = Input(shape=(1, NUM_MFCC, NO_features))

    model = models.get_model_9_rl(input_layer, model_name_prefix='mfcc')

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
                   nb_steps_warmup=args.warmup_steps, gamma=.99, target_model_update=10000,
                   train_interval=4, delta_clip=1.,
                   # train_max_steps=args.max_train_steps
                   )
    dqn.compile('adam', metrics=['mae', 'accuracy'])

    if args.pre_train:
        from feature_type import FeatureType

        datastore: Datastore = None

        if args.pre_train_dataset == DataVersions.IEMOCAP:
            from datastore_iemocap import IemocapDatastore
            datastore = IemocapDatastore(FeatureType.MFCC)

        if args.pre_train_dataset == DataVersions.IMPROV:
            from datastore_improv import ImprovDatastore
            datastore = ImprovDatastore(22)

        if args.pre_train_dataset == DataVersions.SAVEE:
            from datastore_savee import SaveeDatastore
            datastore = SaveeDatastore(FeatureType.MFCC)

        if args.pre_train_dataset == DataVersions.ESD:
            from datastore_esd import ESDDatastore
            datastore = ESDDatastore(FeatureType.MFCC)

        assert datastore is not None

        x_train, y_train, y_gen_train = datastore.get_pre_train_data()

        # dqn.pre_train(x=x_train.reshape((len(x_train), 1, NUM_MFCC, NO_features)), y=y_train,
        #               EPOCHS=args.pretrain_epochs, batch_size=128)

    if args.mode == 'train':
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = f'rl-files/models/dqn_{args.env_name}_weights.h5f'
        checkpoint_weights_filename = 'rl-files/models/dqn_' + args.env_name + '_weights_{step}.h5f'
        log_filename = 'rl-files/logs/dqn_{}_log.json'.format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=100)]

        # if not args.disable_wandb:
        #     wandb_project_name = 'zeta-policy'
        #     callbacks += [WandbLogger(project=wandb_project_name, name=args.env_name)]

        dqn.fit(env, callbacks=callbacks, nb_steps=args.nb_steps, log_interval=10000)

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Finally, evaluate our algorithm for 10 episodes.
        dqn.test(env, nb_episodes=10, visualize=False)

    elif args.mode == 'test':
        weights_filename = f'rl-files/models/dqn_{args.env_name}_weights.h5f'
        if args.weights:
            weights_filename = args.weights
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)


if __name__ == "__main__":
    run()
