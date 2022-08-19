import argparse
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import Input

import models
from SaveBestModelCallback import SaveBestModelCallback
from constants import NUM_MFCC, NO_features, WINDOW_LENGTH, RESULTS_ROOT
from data_versions import DataVersions
from datastore import Datastore
from rl.agents import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint, FileLogger, WandbLogger
from rl.memory import SequentialMemory
from utils import parse_policy, str2dataset, str2bool, get_datastore, get_environment, combine_datastores, store_results

time_str = datetime.now().strftime("%Y_%m_%d_%H_%M")


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='iemocap-rl-v3.1')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--policy', type=str, default='EpsGreedyQPolicy')
    parser.add_argument('--data-version', nargs='+',
                        choices=[DataVersions.IEMOCAP, DataVersions.SAVEE, DataVersions.IMPROV, DataVersions.ESD,
                                 DataVersions.EMODB, DataVersions.KITCHEN_EMODB, DataVersions.KITCHEN_ESD,
                                 DataVersions.KITCHEN_ESD_DB0, DataVersions.KITCHEN_ESD_DBn5,
                                 DataVersions.KITCHEN_ESD_DBn10, DataVersions.KITCHEN_ESD_DBp5,
                                 DataVersions.KITCHEN_ESD_DBp10],
                        type=str2dataset, default=DataVersions.IEMOCAP)
    parser.add_argument('--data-split', nargs='+', type=float, default=None)
    parser.add_argument('--zeta-nb-steps', type=int, default=100000)
    parser.add_argument('--nb-steps', type=int, default=500000)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--pre-train', type=str2bool, default=False)
    parser.add_argument('--pre-train-dataset',
                        choices=[DataVersions.IEMOCAP, DataVersions.IMPROV, DataVersions.SAVEE, DataVersions.ESD,
                                 DataVersions.EMODB],
                        type=str2dataset,
                        default=DataVersions.IEMOCAP)
    parser.add_argument('--pre-train-data-split', type=float, default=None)
    parser.add_argument('--warmup-steps', type=int, default=50000)
    parser.add_argument('--pretrain-epochs', type=int, default=64)
    parser.add_argument('--testing-dataset', type=str2dataset, default=None,
                        choices=[DataVersions.IEMOCAP, DataVersions.IMPROV, DataVersions.SAVEE, DataVersions.ESD,
                                 DataVersions.COMBINED, DataVersions.EMODB, DataVersions.KITCHEN_EMODB,
                                 DataVersions.KITCHEN_ESD, DataVersions.KITCHEN_ESD_DB0, DataVersions.KITCHEN_ESD_DBn5,
                                 DataVersions.KITCHEN_ESD_DBn10, DataVersions.KITCHEN_ESD_DBp5,
                                 DataVersions.KITCHEN_ESD_DBp10])
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--wandb-disable', type=str2bool, default=False, choices=[True, False])
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline'])
    parser.add_argument('--double-dqn', type=str2bool, default=False, choices=[True, False])
    parser.add_argument('--dueling-network', type=str2bool, default=False, choices=[True, False])
    parser.add_argument('--dueling-type', type=str, default='avg', choices=['avg', 'max', 'naive'])
    parser.add_argument('--schedule-csv', type=str, default=None)
    parser.add_argument('--schedule-idx', type=int, default=None)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print("Tensorflow version:", tf.__version__)

    if os.path.exists(f'{RESULTS_ROOT}/{time_str}'):
        raise RuntimeError(f'Results directory {RESULTS_ROOT}/{time_str} is already exists')

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
    tf.compat.v1.experimental.output_all_intermediates(True)
    policy = parse_policy(args)

    data_version_map = {}
    custom_data_split = []
    if args.data_split is not None:
        if len(args.data_split) == 1 and len(args.data_version) > 1:
            for i in range(len(args.data_version)):
                custom_data_split.append(args.data_split[0])
        elif 1 < len(args.data_split) != len(args.data_version) > 1:
            raise RuntimeError("--data-split either should have one value or similar to --data-version")
        else:
            custom_data_split = args.data_split
    else:
        for i in range(len(args.data_version)):
            custom_data_split.append(None)

    if len(args.data_version) == 1:
        target_datastore = get_datastore(data_version=args.data_version[0],
                                         custom_split=None if args.data_split is None else args.data_split[0])
        data_version_map[args.data_version[0]] = target_datastore
        env = get_environment(data_version=args.data_version[0], datastore=target_datastore,
                              custom_split=None if args.data_split is None else args.data_split[0])
    else:
        ds = []
        for i in range(len(args.data_version)):
            d = get_datastore(data_version=args.data_version[i], custom_split=custom_data_split[i])
            data_version_map[args.data_version[i]] = d
            ds.append(d)
        target_datastore = combine_datastores(ds)
        env = get_environment(data_version=DataVersions.COMBINED, datastore=target_datastore, custom_split=None)

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
                   enable_double_dqn=args.double_dqn,
                   enable_dueling_network=args.dueling_network,
                   dueling_type=args.dueling_type)
    # dqn.compile(Adam(learning_rate=.00025), metrics=['mae', 'accuracy'])
    dqn.compile('adam', metrics=['mae', 'accuracy'])

    pre_train_datastore: Datastore = None
    if args.pre_train:

        if args.pre_train_dataset == args.data_version:
            raise RuntimeError("Pre-Train and Target datasets cannot be the same")
        else:
            pre_train_datastore = get_datastore(data_version=args.pre_train_dataset,
                                                custom_split=args.pre_train_data_split)

        assert pre_train_datastore is not None

        (x_train, y_train, y_gen_train), _ = pre_train_datastore.get_data()

        pre_train_log_dir = f'{RESULTS_ROOT}/{time_str}/logs/pre_train'
        if not os.path.exists(pre_train_log_dir):
            os.makedirs(pre_train_log_dir)

        dqn.pre_train(x=x_train.reshape((len(x_train), 1, NUM_MFCC, NO_features)), y=y_train,
                      epochs=args.pretrain_epochs, batch_size=128, log_base_dir=pre_train_log_dir)

    if args.mode == 'train':

        models_dir = f'{RESULTS_ROOT}/{time_str}/models'
        log_dir = f'{RESULTS_ROOT}/{time_str}/logs'

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print(f"Models: {models_dir}")
        # Okay, now it's time to learn something! We capture the interrupt exception so that training
        # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
        weights_filename = f'{models_dir}/dqn_{args.env_name}_weights.h5f'
        checkpoint_weights_filename = models_dir + '/dqn_' + args.env_name + '_weights_{step}.h5f'
        log_filename = log_dir + '/dqn_{}_log.json'.format(args.env_name)
        callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
        callbacks += [FileLogger(log_filename, interval=10)]

        best_model_filename = models_dir + '/best_model_dqn_' + args.env_name + '_weights_{episode}.h5'
        callbacks += [SaveBestModelCallback(model_save_name=best_model_filename)]

        if not args.wandb_disable:
            wandb_project_name = 'zeta-policy'
            wandb_dir = f'{RESULTS_ROOT}/{time_str}/wandb'
            if not os.path.exists(wandb_dir):
                os.makedirs(wandb_dir)
            callbacks += [
                WandbLogger(project=wandb_project_name, name=args.env_name, mode=args.wandb_mode, dir=wandb_dir)]

        dqn.fit(env, callbacks=callbacks, nb_steps=args.nb_steps, log_interval=10000)
        model = dqn.model

        # After training is done, we save the final weights one more time.
        dqn.save_weights(weights_filename, overwrite=True)

        # Testing with Labelled Data
        testing_dataset = args.testing_dataset
        if testing_dataset is not None:
            if testing_dataset == DataVersions.COMBINED:
                if pre_train_datastore is not None:
                    testing_datastore = combine_datastores([target_datastore, pre_train_datastore])
                else:
                    testing_datastore = target_datastore
            else:
                testing_datastore = data_version_map[testing_dataset]
        else:
            # testing dataset is not defined
            if pre_train_datastore is not None:
                testing_datastore = combine_datastores([target_datastore, pre_train_datastore])
            else:
                testing_datastore = target_datastore

        x_test, y_test, _ = testing_datastore.get_testing_data()
        test_loss, test_mae, test_acc, test_mean_q = model.evaluate(
            x_test.reshape((len(x_test), 1, NUM_MFCC, NO_features)), y_test, verbose=1)

        print(f"Test\n\t Accuracy: {test_acc}")

        store_results(f"{log_dir}/results.txt", args=args, experiment=experiment_name, time_str=time_str,
                      test_loss=test_loss, test_acc=test_acc)

        # # Finally, evaluate our algorithm for 10 episodes.
        # dqn.test(env, nb_episodes=10, visualize=False)

    elif args.mode == 'test':
        weights_filename = f'rl-files/models/dqn_{args.env_name}_weights.h5f'
        if args.weights:
            weights_filename = args.weights
        dqn.load_weights(weights_filename)
        dqn.test(env, nb_episodes=10, visualize=True)

    if args.schedule_csv is not None:
        from scheduler_callback import callback
        callback(args.schedule_csv, args.schedule_idx)


if __name__ == "__main__":
    run()
