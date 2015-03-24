import time, math
import numpy as np


def resample(ts, values, num_samples):
    """Convert a list of times and a list of values to evenly spaced samples with linear interpolation"""
    ts = normalize(ts)
    return np.interp(np.linspace(0.0, 1.0, num_samples), ts, values)

def upsample(signal, factor):
    """Increase the sampling rate of a signal (by an integer factor), with linear interpolation"""
    assert type(factor) == int and factor > 1
    result = [None] * ((len(signal) - 1) * factor)
    for i, v in enumerate(signal):
        if i == len(signal) - 1:
            result[-1] = v
            break
        v_ = signal[i+1]
        delta = v_ - v
        for j in range(factor):
            f = (i * factor) + j
            result[f] = v + ((delta / factor) * j)
    return result     

def downsample(signal, factor):
    """Decrease the sampling rate of a signal (by an integer factor), with averaging"""    
    signal = np.array(signal)
    xs = signal.shape[0]
    signal = signal[:xs - (xs % int(factor))]
    result = np.mean(np.concatenate([[signal[i::factor] for i in range(factor)]]), axis=0)
    return result     

def normalize(signal, minimum=None, maximum=None):
    """Normalize a signal to the range 0, 1"""
    signal = np.array(signal).astype('float')
    if minimum is None:
        minimum = np.min(signal)
    signal -= minimum
    if maximum is None:
        maximum = np.max(signal)
    signal /= maximum
    signal = np.clip(signal, 0.0, 1.0)
    return signal    

def threshold(signal, value):
    """Drop all values in a signal to 0 if below the given threshold"""
    signal = np.array(signal)
    return (signal > value) * signal

def limit(signal, value):
    """Limit all values in a signal to the given value"""
    return np.clip(signal, 0, value)

def remove_shots(signal, threshold=None, devs=2, positive_only=False):
    """Replace values in a signal that are above a threshold or vary by a given number of deviations with the average of the surrounding samples"""    
    average = np.average(signal)
    if threshold is not None:
        shot_indexes = [i for (i, sample) in enumerate(signal) if sample > threshold]
    else:
        deviation = np.std(signal)
        shot_indexes = [i for (i, sample) in enumerate(signal) if (sample - average if positive_only else abs(sample - average)) > deviation * devs]
    for i in shot_indexes:
        neighbors = []
        j = i + 1
        k = i - 1
        while j in shot_indexes:
            j += 1
        if j < len(signal):
            neighbors.append(signal[j])
        while k in shot_indexes:
            k -= 1
        if k >= 0:
            neighbors.append(signal[k])
        signal[i] = sum(neighbors) / float(len(neighbors))
    return signal

def compress(signal, value=2.0, normalize=False):
    """Compress the signal by an exponential value (will expand if value<0)"""
    signal = np.array(signal)
    signal = np.power(signal, 1.0 / value)
    return normalize(signal) if normalize else signal

def smooth(signal, size=10, window='blackman'):
    """Apply averaging / low-pass filter to a signal with the given window shape and size"""
    types = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    signal = np.array(signal)
    if size < 3:
        return signal
    s = np.r_[2 * signal[0] - signal[size:1:-1], signal, 2 * signal[-1] - signal[-1:-size:-1]]
    if window == 'flat': # running average
        w = np.ones(size,'d')
    else:
        w = getattr(np, window)(size)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[size - 1:-size + 1]

## todo: def hipass
## smooth and then subtract result    

def detect_peaks(signal, lookahead=300, delta=0):   ## probably a better scipy module...
    """ Detect the local maximas and minimas in a signal
        lookahead -- samples to look ahead from a potential peak to see if a bigger one is coming
        delta -- minimum difference between a peak and surrounding points to be considered a peak (no hills) and makes things faster
        Note: careful if you have flat regions, may affect lookahead
    """    
    signal = np.array(signal)
    peaks = []
    valleys = []
    min_value, max_value = np.Inf, -np.Inf    
    for index, value in enumerate(signal[:-lookahead]):        
        if value > max_value:
            max_value = value
            max_pos = index
        if value < min_value:
            min_value = value
            min_pos = index    
        if value < max_value - delta and max_value != np.Inf:
            if signal[index:index + lookahead].max() < max_value:
                peaks.append([max_pos, max_value])
                drop_first_peak = True
                max_value = np.Inf
                min_value = np.Inf
                if index + lookahead >= signal.size:
                    break
                continue
        if value > min_value + delta and min_value != -np.Inf:
            if signal[index:index + lookahead].min() > min_value:
                valleys.append([min_pos, min_value])
                drop_first_valley = True
                min_value = -np.Inf
                max_value = -np.Inf
                if index + lookahead >= signal.size:
                    break
    return peaks, valleys

def autocorrelate(signal):
    """Get the auto-correlation function of a signal"""    
    x = np.hstack((signal, np.zeros(len(signal))))
    sp = np.fft.rfft(x) 
    tmp = np.empty_like(sp)
    tmp = np.conj(sp, tmp)
    tmp = np.multiply(tmp, sp, tmp)
    ac = np.fft.irfft(tmp)
    ac = np.divide(ac, signal.size, ac)[:signal.size/ 2] 
    tmp = signal.size / (signal.size - np.arange(signal.size / 2, dtype=np.float64)) 
    ac = np.multiply(ac, tmp, ac)
    ac = np.concatenate([ac, np.zeros(signal.size - ac.size)])
    return normalize(ac)

def derivative(signal):
    """Return a signal that is the derivative function of a given signal"""
    def f(x):
        x = int(x)
        return signal[x]
    def df(x, h=0.1e-5):
        return (f(x + h * 0.5) - f(x - h * 0.5)) / h
    return [df(x) for x in xrange(len(signal))]

def integral(signal):
    """Return a signal that is the integral function of a given signal"""
    result = []
    v = 0.0    
    for i in xrange(len(signal)):
        v += signal[i]
        result.append(v)
    return result

def trendline(signal):
    """Returns a  line (slope, intersect) that is the regression line given a series of values."""
    signal = list(signal)
    n = len(signal) - 1
    sum_x = 0
    sum_y = 0
    sum_xx = 0
    sum_xy = 0
    for i in range(1, n + 1):
        x = i
        y = signal[i]
        sum_x = sum_x + x
        sum_y = sum_y + y
        xx = math.pow(x, 2)
        sum_xx = sum_xx + xx
        xy = x*y
        sum_xy = sum_xy + xy
    try:    
        a = (-sum_x * sum_xy + sum_xx * sum_y) / (n * sum_xx - sum_x * sum_x)
        b = (-sum_x * sum_y + n * sum_xy) / (n * sum_xx - sum_x * sum_x)
    except ZeroDivisionError:
        a, b = 0, 0    
    return (b, a) # (slope, intersect)

    
def bandpass_filter(signal, sampling_rate, lowcut, highcut, order=6):
    """In hz"""
    from scipy.signal import butter, lfilter
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    signal = lfilter(b, a, signal)
    return signal
    
def lowpass_filter(signal, sampling_rate, cutoff, order=6):   
    """smooth appears to be much faster in certain situations"""
    from scipy.signal import butter, lfilter 
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    signal = lfilter(b, a, signal)
    return signal

def highpass_filter(signal, sampling_rate, cutoff, order=6):    
    from scipy.signal import butter, lfilter
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    signal = lfilter(b, a, signal)
    return signal    