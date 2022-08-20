import os

import pandas as pd
from sklearn.metrics import recall_score

from constants import EMOTIONS
from rl.callbacks import Callback


class SaveBestModelCallback(Callback):

    def __init__(self, model_save_name: str, model_dir: str, ):
        self.step_inferences = []
        self.best_uar = 0
        self.model_save_name = model_save_name
        self.model_dir = model_dir

    def on_episode_begin(self, episode, logs):
        self.step_inferences = []

    def on_episode_end(self, episode, logs):
        df = pd.DataFrame(self.step_inferences)

        UAR = recall_score(df['ground_truth'].to_numpy(), df['inference'].to_numpy(), average='macro')

        if UAR > self.best_uar:
            self.best_uar = UAR
            save_dir = f'{self.model_dir}/{str(episode)}'
            os.makedirs(save_dir)
            self.model.model.save(save_dir + "/" + self.model_save_name)

        del df

    def on_step_end(self, step, logs):
        self.step_inferences.append({
            'ground_truth': EMOTIONS[int(logs['info']['ground_truth'])],
            'inference': EMOTIONS[int(logs['action'])]
        })
