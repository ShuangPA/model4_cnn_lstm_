model_name                = "model4_cnn_LSTM_Z0"

path_work                 = f"work.{model_name}"
path_model                = f"{path_work}/model"

max_frame_num             = 200
mfcc_num                  = 20
sample_rate               = 16000

epoch_num                 = 100
batch_size                = 2048
neg_sample_ratio          = 15
rnn_layer_num             = 1
rnn_unit_num              = 100
dropout_keep_prob         = 0.5
attention_num             = 1

evaluate_freq             = 100_000 // batch_size

result_path               = "./resultZ0"
train_file                = "./dataset/version.10.train.pydict"
test_file                 = "./dataset/version.10.test.pydict"
wav_folder                = "./toefl_voice_data"
