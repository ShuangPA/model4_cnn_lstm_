#!/usr/bin/env python3
import sys, os
sys.path.append(os.environ["HOME"] + '/asr_context/insight_nlp')
sys.path.append(os.environ["HOME"] + '/asr_context/filler_z')
from model4_cnn_lstm import *
from model4_cnn_lstm._data import DataSet
from model4_cnn_lstm._model import Model
from model4_cnn_lstm import model_config
from model4_cnn_lstm.predict import Predictor


def main():
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  preditor = Predictor()
  preditor.load_specific_model(34999)
  data_set = DataSet(model_config.test_file)
  all_true_label, all_pred_label, all_pred_prob  = \
    preditor.predict_dataset(data_set)

  print(f"done!")
if __name__ == '__main__':
  main()