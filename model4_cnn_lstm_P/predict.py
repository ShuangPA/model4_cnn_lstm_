#!/usr/bin/env python3
import sys, os
sys.path.append(os.environ["HOME"] + '/asr_context/insight_nlp')
from model4_cnn_lstm_P import *
from model4_cnn_lstm_P._data import DataSet
from model4_cnn_lstm_P._model import Model
from model4_cnn_lstm_P import model_config

class Predictor(object):
  def __init__(self):
    self._graph = tf.Graph()
    with self._graph.as_default():
      self._model = Model(
        max_seq_len=model_config.max_frame_num,
        mfcc_num=model_config.mfcc_num,
        rnn_layer_num=model_config.rnn_layer_num,
        rnn_unit_num=model_config.rnn_unit_num,
        attention_num=model_config.attention_num,
        neg_sample_weight=1,
        is_training=False,
        T_embedding_size=model_config.T_embedding_dim,
        T_filter_sizes=model_config.T_filter_sizes,
        T_num_filters=model_config.T_num_filters,
      )

    self._sess = tf.Session(graph=self._graph)

  def load_model(self, model_path: str):
    print(f"loading model from '{model_path}'")
    with self._graph.as_default():
      tf.train.Saver().restore(
        self._sess,
        tf.train.latest_checkpoint(model_path)
      )

  def load_specific_model(self, model_path):
    print(f"loading model from: '{model_path}'")
    with self._graph.as_default():
      tf.train.Saver().restore(
        self._sess,
        model_path
      )

  def predict_dataset(self, data_set: DataSet, eval_precision: bool=True):
    print("-" * 80)
    time_start = time.time()

    model = self._model
    batch_iter = data_set.creat_batches(32, 1, False)
    all_true_label = []
    all_pred_label = []
    all_pred_prob = []
    for batch_id, [feats, label, name, phonemes_info] in enumerate(batch_iter):
      pred_y, pred_prob = self._sess.run(
        fetches=[
          model.predicted_class,
          model.class_probs,
        ],
        feed_dict={
          model.input_x: feats,
          model.input_y: label,
          model.input_phonemes: phonemes_info,
        }
      )

      all_true_label.extend(label)
      all_pred_label.extend(pred_y)
      all_pred_prob.extend([p1 for p0, p1 in pred_prob])

    if eval_precision:
      eval = Measure.calc_precision_recall_fvalue(all_true_label,
                                                  all_pred_label)
      print(f"evaluate({data_set.feat_file}): "
            f"#sample: {len(all_true_label)} {eval[1]}")

      duration = time.time() - time_start
      print(f"evaluation takes {duration:.2f} seconds.")
      print("-" * 80)

      #return all_true_label, all_pred_label, all_pred_prob
      return (f"evaluate({data_set.feat_file}): "
              f"#sample: {len(all_true_label)} {eval[1]}")

def main():
  parser = optparse.OptionParser(usage="cmd [optons] feat_folder1 ...")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")
  (options, args) = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  preditor = Predictor()
  #preditor.load_model(model_config.path_model)
  preditor.load_specific_model('./model4_cnn_lstm/original.work.model4_cnn_LSTM/model/asr-35879')
  data_sets = [DataSet(feat_folder) for feat_folder in args]
  for data_set in data_sets:
    preditor.predict_dataset(data_set)

if __name__ == '__main__':
  main()

