#!/usr/bin/env python3
#Modify by shuangzhao
import sys, os
sys.path.append(os.environ["HOME"] + '/asr_context/insight_nlp')
sys.path.append(os.environ["HOME"] + '/asr_context/filler_z')
from model4_cnn_lstm_0 import *
from model4_cnn_lstm_0._data import DataSet
from model4_cnn_lstm_0._model import Model
from model4_cnn_lstm_0.predict import Predictor
from model4_cnn_lstm_0 import model_config

class Trainer:
  def __init__(self):
    random.seed()
    self._data_train = DataSet(model_config.train_file)
    self._data_tests = [self._data_train, DataSet(model_config.test_file)]

    nlp.ensure_folder_exists(model_config.path_work, True)
    nlp.ensure_folder_exists(model_config.path_model, True)

    # load model
    self._model = Model(
      max_seq_len=model_config.max_frame_num,
      mfcc_num=model_config.mfcc_num,
      rnn_layer_num=model_config.rnn_layer_num,
      rnn_unit_num=model_config.rnn_unit_num,
      attention_num=model_config.attention_num,
      neg_sample_weight=1 / model_config.neg_sample_ratio,
      is_training=True,
    )

    self._predictor = Predictor()
    self._optimizer_op = TF.construct_optimizer(self._model.loss)

  def _evaluate(self, batch_id):
    # save models
    print(f"saving model[{batch_id}] ...")
    self._saver.save(
      self._sess, os.path.join(f"{model_config.path_model}/asr"),
      global_step=batch_id,
    )

    self._predictor.load_model(model_config.path_model)
    data_output = ""
    for data_set in self._data_tests:
      print_flush(f"evaluate[{batch_id}]")
      data_output += f"batch_id:{batch_id} --- " \
                     + (self._predictor.predict_dataset(data_set)) + "\n"
    return data_output

  def train(self):
    self._sess = tf.Session()
    self._sess.run(tf.global_variables_initializer())
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
    model = self._model

    batch_iter = self._data_train.creat_batches(
      model_config.batch_size, model_config.epoch_num, True
    )
    batch_id = 0
    outputs = ""
    for batch_id, [feats, label, name] in enumerate(batch_iter):

      time_start = time.time()
      train_loss, error_count, _ = self._sess.run(
        fetches=[
          model.loss,
          model.error_count,
          self._optimizer_op
        ],
        feed_dict={
          model.input_x: feats,
          model.input_y: label,
          model.dropout_keep_prob: model_config.dropout_keep_prob,
        }
      )
      duration = time.time() - time_start
      print_flush(
        f"batch: {batch_id} "
        f"loss: {train_loss:.4f} #error: {error_count} time: {duration:.3f}"
      )

      if (batch_id + 1) % model_config.evaluate_freq == 0:
        outputs += self._evaluate(batch_id)

    outputs += self._evaluate(batch_id)
    with open(model_config.result_path, 'w') as f:
      f.write(outputs)
    print("DONE!")

def main():
  parser = optparse.OptionParser(usage="cmd [optons] ..]")
  # parser.add_option("-q", "--quiet", action="store_true", dest="verbose",
  parser.add_option("--gpu", default="-1", help="default=-1")

  # default=False, help="")
  (options, args) = parser.parse_args()
  print(options)
  os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

  Trainer().train()

if __name__ == '__main__':
  main()
