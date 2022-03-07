import librosa.display
import mir_eval.melody
import scipy.io.wavfile

from training import *
import pathlib
import matplotlib.pyplot as plt


def show_example_plot(file="./run 20211127 010004/mdl_20211127 022433_tanh_2x4(bi).pt",
                      features=None,
                      test_key="ah_development_guitar_Guitar_Licks_06-11",
                      figsize=(4.8, 4.8)):
    if features is None:
        features = ['rcd', 'superflux']
    mgr = ModelManager(datasets.BockSet(), ModelConfig(features=features), TrainingConfig(),
                       load_file=file)
    test_output(mgr, test_key, figsize=figsize)


def generate_demo_audio(model_file="./run 20211127 010004/mdl_20211127 022433_tanh_2x4(bi).pt",
                        features=None,
                        wav=None,
                        sr=44100,
                        onset_list=None,
                        save_to="./demo_audio/",
                        height=0.35):
    boeck_set = datasets.BockSet()
    if features is None:
        features = ['rcd', 'superflux']
    if wav is None:
        wav = boeck_set.get_piece("ah_development_guitar_Guitar_Licks_06-11").get_wave()
        sr = 44100
        onset_list = boeck_set.get_piece("ah_development_guitar_Guitar_Licks_06-11").get_onsets_seconds()
    pathlib.Path(save_to).mkdir(parents=True, exist_ok=True)
    # Plot wave
    plt.figure(figsize=(15, 5))
    librosa.display.waveplot(wav, sr, alpha=0.8)
    # for x in onset_list:
    #     plt.axvline(x=x, color='red', alpha=0.3)
    plt.show()
    # Plot spectrogram
    plt.figure(figsize=(15, 5))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav, hop_length=441)))
    librosa.display.specshow(D, y_axis='log', sr=sr, hop_length=441,
                             x_axis='time')
    plt.show()
    # Get onset list
    if onset_list is None:
        mgr = ModelManager(boeck_set, ModelConfig(features=features), TrainingConfig(),
                           load_file=model_file)
        onset_list = mgr.predict_onsets_offline(wave=wav, height=height)
        onset_list = onset_utils.combine_onsets_avg(onset_list, 0.030)
    # Save wave
    scipy.io.wavfile.write(save_to + "/original.wav", sr, wav)
    # Generate audio with onset clicks
    clicks = mir_eval.sonify.clicks(onset_list, sr, length=len(wav))
    wav_and_clicks = wav + clicks
    # Normalize to [-1, 1] and set dtype for saving in 32bit float PCM
    wav_and_clicks = utils.normalize_wav(wav_and_clicks)
    scipy.io.wavfile.write(save_to + "/with_clicks.wav", sr, wav_and_clicks)


def generate_onset_demo_audio(wav,
                              model_file="./run 20211127 010004/mdl_20211127 022433_tanh_2x4(bi).pt",
                              features=None,
                              sr=44100,
                              save_to="./demo_audio_content/",
                              height=0.35):
    if features is None:
        features = ['rcd', 'superflux']
    boeck_set = datasets.BockSet()
    mgr = ModelManager(boeck_set, ModelConfig(features=features), TrainingConfig(),
                       load_file=model_file)
    onset_list = mgr.predict_onsets_offline(wave=wav, height=height)
    onset_list = onset_utils.combine_onsets_avg(onset_list, 0.030)
    pathlib.Path(save_to).mkdir(parents=True, exist_ok=True)
    scipy.io.wavfile.write(save_to + "/original.wav", sr, wav)
    # Generate audio with onset clicks
    clicks = mir_eval.sonify.clicks(onset_list, sr, length=len(wav))
    wav_and_clicks = wav + clicks
    # Normalize to [-1, 1] and set dtype for saving in 32bit float PCM
    wav_and_clicks = utils.normalize_wav(wav_and_clicks)
    scipy.io.wavfile.write(save_to + "/with_clicks.wav", sr, wav_and_clicks)


if __name__ == '__main__':
    # wav, sr = librosa.load(r"E:\Documents\Personal\收藏\音乐\YouTube Music\Cat_life.mp3",
    #                        sr=44100)
    # wav = wav / 2
    # generate_onset_demo_audio(wav)
    # fs = 44100

    # click_high = np.sin(2 * np.pi * np.arange(fs * .1) * 1046.5 / (1. * fs))
    # click_high *= np.exp(-np.arange(fs * .1) / (fs * .01))
    #
    # click_low = np.sin(2 * np.pi * np.arange(fs * .1) * 784.0 / (1. * fs))
    # click_low *= np.exp(-np.arange(fs * .1) / (fs * .01))
    #
    # scipy.io.wavfile.write("./click_high.wav", fs, click_high)
    # scipy.io.wavfile.write("./click_low.wav", fs, click_low)
    generate_demo_audio()
