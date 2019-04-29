
from collections import defaultdict
from operator import itemgetter
from pa_nlp.audio.acoustic_feature_tf import DataGraphMFCC
from pa_nlp.audio.audio_helper import AudioHelper
from pa_nlp.common import print_flush
from pa_nlp.measure import Measure
import numpy
import optparse
import os
import pa_nlp.common as nlp
import pa_nlp.tensorflow as TF
import random
import tensorflow as tf
import time
import typing
