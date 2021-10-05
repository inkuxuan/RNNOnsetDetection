import numpy as np
import matplotlib.pyplot as plt


def plot_wave(wave):
    fig = plt.figure()
    fig.set_figwidth(20)
    fig.set_figheight(2)
    plt.plot(wave, color=(41 / 255., 104 / 255., 168 / 255.))
    fig.axes[0].set_xlim([0, len(wave)])
    fig.axes[0].set_xlabel('sample')
    fig.axes[0].set_ylabel('amplitude')
    plt.show()


def plot_music_net_roll(labels_, sr, start=0, end=30):
    stride = 512  # 512 samples between windows
    wps = sr / float(stride)  # ~86 windows/second
    Yvec = np.zeros((int((end - start) * wps + 1), 128))  # 128 distinct note labels
    colors = {41: .33, 42: .66, 43: 1}

    for window in range(int(start * wps), int(end * wps)):
        labels = labels_[window * stride]
        for label in labels:
            Yvec[window - int(start * wps), label.data[1]] = label.data[0] / 128
    fig = plt.figure(figsize=(20, 5))
    plt.imshow(Yvec.T, aspect='auto', cmap='ocean_r')
    plt.gca().invert_yaxis()
    fig.axes[0].set_xlabel('window')
    fig.axes[0].set_ylabel('note (MIDI code)')
    plt.show()


def plot_odf(odf, title="ODF", onset_target=None, onsets=None):
    r"""

    :param odf: Onset Detection Function, with frames in the first axis
    :param title: Title of the figure
    :param onset_target: Optional overlapping function
    :param onsets: Onsets, in list of frame numbers, will be plotted as vertical lines
    :return:
    """
    fig = plt.figure()
    plt.plot(odf)
    if onset_target is not None:
        plt.plot(onset_target, color='red', alpha=0.3)
    if onsets is not None:
        for x in onsets:
            plt.axvline(x=x, color='red', alpha=0.3)
    plt.title(title)
    plt.show()


def test_with_key(key, start, stop):
    import datasets
    music_net = datasets.MusicNet()
    example_key = key
    piece = music_net.get_piece(example_key)
    wave, onsets, sr = piece.get_data()
    plot_music_net_roll(piece.get_raw_labels(), sr, start=start, end=stop)


if __name__ == '__main__':
    test_with_key('2297', 15, 30)
