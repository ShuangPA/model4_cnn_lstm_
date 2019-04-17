from model4_cnn_lstm_0 import *
from model4_cnn_lstm_0 import model_config

class DataSet:
  def __init__(self, feat_file):
    self._data = None
    self.feat_file = feat_file
    self._file_buff = {}
    self._file_durations = {}
    self._data_graph_mfcc = DataGraphMFCC(model_config.sample_rate,
                                          model_config.mfcc_num)

  def _get_feature(self, sample: dict):
    file_id = sample["file"]
    if file_id not in self._file_buff:
      file_name = os.path.join(model_config.wav_folder, f"{file_id}.wav")
      wav_file = AudioHelper.convert_to_standard_wav(file_name)
      audio = self._data_graph_mfcc.read_16bits_wav_file(wav_file)
      duration = len(audio) / model_config.sample_rate    # in seconds.
      self._file_durations[file_id] = duration

      feat, _ = self._data_graph_mfcc.calc_feats(audio)
      self._file_buff[file_id] = feat

    feat = self._file_buff[file_id]
    feat_len = len(feat)
    range_f, range_t = sample["time_range"]
    range_f, range_t = range_f / 100, range_t / 100

    file_len = self._file_durations[file_id]
    index_f = int(feat_len / file_len * range_f)
    index_t = int(feat_len / file_len * range_t)

    data = feat[index_f: index_t][: model_config.max_frame_num]
    diff = model_config.max_frame_num - len(data)
    if diff > 0:
      extra = numpy.array([0.] * 3 * model_config.mfcc_num * diff,
                          numpy.float32)
      extra = extra.reshape([diff, model_config.mfcc_num, 3])
      return numpy.concatenate([data, extra], axis=0)
    else:
      assert diff == 0
      return data

  def _clean_buff(self, batch_data: list):
    file_ids = set([d["file"] for d in batch_data])
    file_buff = dict()
    for file_id, data in self._file_buff.items():
      if file_id in file_ids:
        file_buff[file_id] = data

    del self._file_buff
    self._file_buff = file_buff

  def creat_batches(self, batch_size: int, epoch_num: int, shuffle: bool):
    if self._data is None:
      file_id_to_samples = defaultdict(list)
      for idx, ln in enumerate(open(self.feat_file)):
        if idx % 1000 == 0:
          nlp.print_flush(f"{self.feat_file}: {idx}")

        d = eval(ln)
        file_id_to_samples[d["file"]].append(d)

      self._data = list(file_id_to_samples.items())

    for epoch_id in range(epoch_num):
      if shuffle:
        random.shuffle(self._data)
        for _, samples in self._data:
          random.shuffle(samples)

      samples = sum([pair[1] for pair in self._data], [])

      next = iter(samples)
      _ = range(batch_size)
      while True:
        batch_samples = list(map(itemgetter(1), zip(_, next)))
        if batch_samples == []:
          break

        data = [[self._get_feature(d), d["label"], d["file"]]
                for d in batch_samples]
        feats, labels, names = nlp.split_to_sublist(data)
        self._clean_buff(batch_samples)
        yield feats, labels, names

      print(f"The '{self.feat_file}' {epoch_id + 1}/{epoch_num} "
            f"epoch has finished!")
