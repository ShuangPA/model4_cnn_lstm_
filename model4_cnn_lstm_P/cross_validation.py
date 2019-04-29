import sys, os, re
sys.path.append(os.environ["HOME"] + '/asr_context/insight_nlp')
sys.path.append(os.environ["HOME"] + '/asr_context/filler_z')
from model4_cnn_lstm import *
from model4_cnn_lstm._data import DataSet
from model4_cnn_lstm._model import Model
from model4_cnn_lstm import model_config
from model4_cnn_lstm.predict import Predictor
from model4_cnn_lstm.train import Trainer

def make_data(num_folder, file_path, output_path):
  lines = open(file_path, 'r').readlines()
  total_len = len(lines)
  for i in range(num_folder):
    dev_list = lines[int(i*total_len/num_folder) : int((i+1)*total_len/num_folder)]
    train_list = lines[ : int(i*total_len/num_folder)] + lines[int((i+1)*total_len/num_folder) : ]
    fp = output_path + f'/{i}'
    if not os.path.exists(fp):
      os.mkdir(fp)
    f = open(fp + '/train.pydict', 'w')
    for line in train_list:
      f.write(line)
    f2 = open(fp + '/dev.pydict', 'w')
    for line in dev_list:
      f2.write(line)
    print(i)
    print(len(train_list))
    print(len(dev_list))
    print('*'*80)

def train_all(num_folder):
  i = num_folder
  trainer = Trainer(f'../dataset/{i}/train.pydict', f'../dataset/{i}/dev.pydict', f'work{i}/model')
  trainer.train(f'./tem_output/train_output{i}')

def predict_and_compare(num_folder, dev_file_path):
  def _find_best_id(filename):
    lines = open(filename, 'r').readlines()
    id, max_f = 0, 0
    for line in lines:
      if 'dev.pydict' in line:
        f_score = float(re.findall(r"\'f\'\:(.*)\}", line)[0])
        if f_score > max_f:
          max_f = f_score
          id = int(re.findall(r"id\:(.*)\s---", line)[0])
    return id, max_f

  def _read_truth_file(file_name: str):
    info_needed = []
    lines = open(file_name, 'r').readlines()
    for line in lines:
      info_dict = eval(line)
      info_needed.append([info_dict['file'], info_dict['time_range'],
                          info_dict['filler']])
    return info_needed

  i = num_folder
  batch_id, f_score = _find_best_id(f'./tem_output/train_output{i}')
  print(f"choose batch: {batch_id}, f_score is: {f_score}")
  predictor = Predictor()
  predictor.load_specific_model(f'./work{i}/model/asr-{batch_id}')
  dev_info = _read_truth_file(dev_file_path + f'/{i}/dev.pydict')
  data_set = DataSet(dev_file_path + f'/{i}/dev.pydict')
  all_true_label, all_pred_label, all_pred_prob = \
    predictor.predict_dataset(data_set)
  f1 = open(f'./cross_vali_output/set{i}_truth0_pred1.txt', 'w')
  f2 = open(f'./cross_vali_output/set{i}_truth1_pred0.txt', 'w')
  f1.write('true_label pred_label pred_prob file time_range filler\n')
  f2.write('true_label pred_label pred_prob file time_range filler\n')
  for idx in range(len(all_true_label)):
    if int(all_true_label[idx]) == 0 and int(all_pred_label[idx]) == 1:
      f1.write(f"{all_true_label[idx]} {all_pred_label[idx]} {all_pred_prob[idx]}"
            f" {dev_info[idx][0]} {dev_info[idx][1]} {dev_info[idx][2]}"
            f"\n")
    if int(all_true_label[idx]) == 1 and int(all_pred_label[idx]) == 0:
      f2.write(f"{all_true_label[idx]} {all_pred_label[idx]} {all_pred_prob[idx]}"
            f" {dev_info[idx][0]} {dev_info[idx][1]} {dev_info[idx][2]}"
            f"\n")
  print(f"writing result for set {i} is done")


def main():
  parser = optparse.OptionParser(usage="cmd [optons] feat_folder1 ...")
  parser.add_option("--gpu", default="-1", help="default=-1")
  parser.add_option("--set", default=0, type=int , help="which set of data")
  (options, args) = parser.parse_args()
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  #make_data(5, '../dataset/train.pydict', '../dataset')
  #train_all(options.set)
  predict_and_compare(options.set, '../dataset')

if __name__ == '__main__':
  main()