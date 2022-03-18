import scipy.io.wavfile
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal as sp

data = glob.glob('covid/posCov/*.wav')
Data=[]
P_mfcc=[]
P_zcrossing= []
P_centroid=[]
P_srolloff=[]
P_flatness=[]
for path_i in range(len(data)):
    sample_rate, signal = scipy.io.wavfile.read(data[path_i])
    signal = signal[0:int(2 * sample_rate)]
    signal = np.float32(signal)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    zcrossing = librosa.feature.zero_crossing_rate(y=signal)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    srolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)
    flatness = librosa.feature.spectral_flatness(y=signal)
    P_mfcc.append(np.asarray(mfccs).flatten())
    P_zcrossing.append(np.asarray(zcrossing).flatten())
    P_centroid.append(np.asarray(centroid).flatten())
    P_srolloff.append(np.asarray(srolloff).flatten())
    P_flatness.append(np.asarray(flatness).flatten())

P_mfcc=np.asarray(P_mfcc)
P_zcrossing=np.asarray(P_zcrossing)
P_centroid=np.asarray(P_centroid)
P_srolloff=np.asarray(P_srolloff)
P_flatness=np.asarray(P_flatness)


data = glob.glob('covid/negCov/*.wav')
Data=[]
N_mfcc=[]
N_zcrossing= []
N_centroid=[]
N_srolloff=[]
N_flatness=[]
for path_i in range(len(data)):
    sample_rate, signal = scipy.io.wavfile.read(data[path_i])
    signal = signal[0:int(2 * sample_rate)]
    signal = np.float32(signal)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    mfccs = np.mean(mfccs.T, axis=0)
    zcrossing = librosa.feature.zero_crossing_rate(y=signal)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    srolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate)
    flatness = librosa.feature.spectral_flatness(y=signal)
    N_mfcc.append(np.asarray(mfccs).flatten())
    N_zcrossing.append(np.asarray(zcrossing).flatten())
    N_centroid.append(np.asarray(centroid).flatten())
    N_srolloff.append(np.asarray(srolloff).flatten())
    N_flatness.append(np.asarray(flatness).flatten())

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
N_mfcc=np.asarray(N_mfcc )
N_zcrossing=np.asarray(N_zcrossing)
N_centroid=np.asarray(N_centroid)
N_srolloff=np.asarray(N_srolloff)
N_flatness=np.asarray(N_flatness)


allMfcc= np.concatenate((P_mfcc ,N_mfcc))
allzCrossing=np.concatenate((P_zcrossing,N_zcrossing), axis=None)
allCentroid=np.concatenate((P_centroid,N_centroid), axis=None)
allSrolloff=np.concatenate((P_srolloff,N_srolloff), axis=None)
allFlatness=np.concatenate((P_flatness,N_flatness), axis=None)


print(signal)
print(sample_rate)
print(P_mfcc.size)
print((N_mfcc.size))
print(allMfcc.size)

x = np.arange(1)
target1=x.repeat(117)
#target=list(target)
# target1=np.where(target==1, 'N', target)
x = np.arange(1)+1
target2=x.repeat(91)
# target2=np.where(target==2, 'MVP', target)

target=np.concatenate((target1, target2))





