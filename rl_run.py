import argparse
from datetime import datetime

import gym
import os
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

import models
from constants import NUM_MFCC, NO_features, WINDOW_LENGTH, RESULTS_ROOT
from data_versions import DataVersions
from datastore import Datastore
from environments import IemocapEnv, SaveeEnv, ImprovEnv, ESDEnv
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, WandbLogger
from rl.memory import SequentialMemory
from utils import parse_policy, str2dataset, str2bool


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='iemocap-rl-v3.1')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--policy', type=str, default='EpsGreedyQPolicy')
    parser.add_argument('--data-version',
                        choices=[DataVersions.IEMOCAP, DataVersions.SAVEE, DataVersions.IMPROV, DataVersions.ESD],
                        type=str2dataset, default=DataVersions.IEMOCAP)
    parser.add_argument('--zeta-nb-steps', type=int, default=100000)
    parser.add_argument('--nb-steps', type=int, default=500000)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--pre-train', type=str2bool, default=False)
    parser.add_argument('--pre-train-dataset',
                        choices=[DataVersions.IEMOCAP, DataVersions.IMPROV, DataVersions.SAVEE, DataVersions.ESD],
                        type=str2dataset,
                        default=DataVersions.IEMOCAP)
    parser.add_argument('--warmup-steps', type=int, default=50000)
    parser.add_argument('--pretrain-epochs', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--wandb-disable', type=str2bool, default=False, choices=[True, False])
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Tensorflow version:", tf.__version__)

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

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
                   train_interval=4, delta_clip=1.)
    dqn.compile(Adam(learning_rate=.00025), metrics=['mae', 'accuracy'])

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

        # x_train, y_train, y_gen_train = datastore.get_pre_train_data()

        # dqn.pre_train(x=x_train.reshape((len(x_train), 1, NUM_MFCC, NO_features)), y=y_train,
        #               EPOCHS=args.pretrain_epochs, batch_size=128)

    if args.mode == 'train':
        time_str = datetime.now().strftime("%Y_%m_%d_%H_%M")

        models_dir = f'{RESULTS_ROOT}/{time_str}/models'
        log_dir = f'{RESULTS_ROOT}/{time_str}/logs'

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        print(f"Models: {models_dir}")
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = f'{models_dir}/dqn_{args.env_name}_weights.h5f'
        checkpoint_weights_filename = models_dir + '/dqn_' + args.env_name + '_weights_{step}.h5f'
        log_filename = log_dir + '/dqn_{}_log.json'.format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=10)]

        if not args.wandb_disable:
            wandb_project_name = 'zeta-policy'
            wandb_dir = f'{RESULTS_ROOT}/{time_str}/wandb'
            if not os.path.exists(wandb_dir):
                os.makedirs(wandb_dir)
            callbacks += [
                WandbLogger(project=wandb_project_name, name=args.env_name, mode=args.wandb_mode, dir=wandb_dir)]

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
