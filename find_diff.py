#!/usr/bin/env python3
import sys, os
sys.path.append(os.environ["HOME"] + '/asr_context/insight_nlp')
sys.path.append(os.environ["HOME"] + '/asr_context/filler_z')
from model4_cnn_lstm import *
from model4_cnn_lstm._data import DataSet
from model4_cnn_lstm._model import Model
from model4_cnn_lstm import model_config
from model4_cnn_lstm.predict import Predictor

def read_train_file(file_name: str):
  info_needed = []
  lines = open(file_name, 'r').readlines()
  for line in lines:
    info_dict = eval(line)
    info_needed.append([info_dict['file'], info_dict['time_range'],
                        info_dict['filler']])
  return info_needed

def main():
  parser = optparse.OptionParser(usage="cmd [optons] feat_folder1 ...")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")
  (options, args) = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  preditor = Predictor()
  #preditor.load_model(model_config.path_model)
  preditor.load_specific_model(34999)
  data_set = DataSet(model_config.train_file)
  all_true_label, all_pred_label, all_pred_prob  = \
    preditor.predict_dataset(data_set)
  train_info = read_train_file(model_config.train_file)
  with open('./tem_output/train.compare.txt', 'w') as f:
    f.write('true_label pred_label pred_prob file time_range filler\n')
    for idx in range(len(all_true_label)):
      f.write(f"{all_true_label[idx]} {all_pred_label[idx]} {all_pred_prob[idx]}"
              f" {train_info[idx][0]} {train_info[idx][1]} {train_info[idx][2]}"
              f"\n")
  print(f"total length: {len(all_true_label)}")
  print(f"output file write is done!")
if __name__ == '__main__':
  main()

