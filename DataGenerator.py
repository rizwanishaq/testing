import os
import numpy as np
from glob import glob
from math import ceil
from utils import sparse_tuple_from as sparse_tuple_from
from utils import pad_sequences as pad_sequences
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
from scipy.signal import spectrogram
from text import text_to_char_array, normalize_txt_file


def preprocess_audio_spectrogram(audio_filename):
    fs, s = wav.read(audio_filename)
    if s.ndim>1:
        s = s[:,0]
    s = s/np.max(np.abs(s)).astype('float')
    if fs!=8000:
        s = resample_poly(s,8000,fs)
        s = s/max(abs(s)).astype('float')
    # Spectrogram Calculation
    _,_, Sxx = spectrogram(s,fs=8000, nfft=1024)
    return Sxx


def _input_data(wavfiles):

    textfiles = [file.replace('.wav','.txt') for file in wavfiles]
    audio = []
    audio_len = []
    transcript = []
    transcript_len = []

    for target_filename, audio_filename in zip(textfiles, wavfiles):

        Sxx = preprocess_audio_spectrogram(audio_filename)
        inputs_data = 20*np.log10(Sxx).T.astype('float32')
        inputs_data = (inputs_data-np.mean(inputs_data,axis=0))/np.std(inputs_data, axis=0)
        audio.append(inputs_data)
        audio_len.append(np.int32(len(inputs_data)))
        # Readings targets
        # load text transcription and convert to numerical array
        targets = normalize_txt_file(target_filename)
        targets = text_to_char_array(targets)
        transcript.append(targets)
        transcript_len.append(len(targets))

    audio = np.asarray(audio)
    audio_len = np.asarray(audio_len)
    transcript = np.asarray(transcript)
    transcript_len = np.asarray(transcript_len)


    train_inputs, train_seq_len = pad_sequences(audio)
    # Creating sparse representation to feed the placeholder
    train_targets = sparse_tuple_from(transcript)


    return train_inputs, train_targets, train_seq_len





class DataGenerator():

    def __init__(self, data_dir, batch_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.audiofiles = glob(os.path.join(self.data_dir, '*.wav'))
        self.num_batchs = ceil(len(self.audiofiles)/batch_size)

    def __len__(self):
        return len(self.audiofiles)


    def next_batch(self):
        for batch in range(self.num_batchs):
            audio_files = self.audiofiles[batch*self.batch_size:min((batch+1)*self.batch_size, len(self.audiofiles))]
            train_inputs, train_targets, train_seq_len = _input_data(audio_files)
            yield train_inputs, train_targets, train_seq_len



if __name__ == '__main__':
    data_dir = './wav_folder'
    dataclass = DataGenerator(data_dir, 2)
    print(len(dataclass))
    for i in range(2):
        for (train_inputs, train_targets, train_seq_len) in dataclass.next_batch():
            pass
            #print(train_inputs.shape, train_seq_len, train_targets)
