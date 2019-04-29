model_name                = "model4_cnn_LSTM_TEST4"

path_work                 = f"work.{model_name}"
path_model                = f"{path_work}/model"

max_frame_num             = 200
mfcc_num                  = 20
sample_rate               = 16000

epoch_num                 = 100
batch_size                = 2048
neg_sample_ratio          = 10
rnn_layer_num             = 1
rnn_unit_num              = 100
dropout_keep_prob         = 0.5
attention_num             = 1

evaluate_freq             = 100_000 // batch_size

result_path               = "./resultTEST4"
train_file                = "./dataset/T_train2.pydict"
test_file                 = "./dataset/T_version.09.test.pydict"
phoneme_file              = "./dataset/T_train2.pydict"
wav_folder                = "./toefl_voice_data"


T_embedding_dim           = 128
T_filter_sizes            = [3,4,4,5,5,6]
T_num_filters             = 128

# 0: [2,3,4]
# 1: [2,3,4,4,5,6]
# 2: [4,5,6]
# 3: [4,5,6,7,8,9]
# 4: [3,4,4,5,5,6]
