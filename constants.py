import socket

host = socket.gethostname()

# environment specific constants

PKL_ROOT = './pkl/'

if host == 'asimov':
    PKL_ROOT = '/home/u1116888/projects/iemocap_dataset/'

if host == 'Thejans-MacBook-Pro.local':
    PKL_ROOT = '/Users/jayaneetha/PycharmProjects/iemocap_dataset/pkl/'

if host == 'thejanr-u20dt':
    PKL_ROOT = '/home/jayaneetha/iotheath/iemocap_dataset/'

EMOTIONS = ['hap', 'sad', 'ang', 'neu']
GENDERS = ['M', 'F']
NUM_MFCC = 40
DURATION = 2
SR = 22050
NO_features = 87  # sr22050&2sec
# NO_features = 173  # sr22050&4sec
# NO_features = 63  # sr8000&4sec
# NO_features = 63  # sr16000&2sec
# NO_features = 251  # sr16000&8sec
WINDOW_LENGTH = 1
