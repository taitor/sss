import os
import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft
from scipy import ifft
from scipy.io.wavfile import read, write

def main():
  args = get_args()
  input_file_name = args['file']
  k = int(args['k'])
  method = args['method']

  fs, data = read(input_file_name)
  data = data[:int(data.shape[0] / 12)]

  fftLen = 512
  win = hamming(fftLen)
  step = int(fftLen / 4)

  w = STFT(data, win, step)

  real = w.real
  imag = w.imag

  angle = w / (np.abs(w) + 1e-15)

  power = real * real + imag * imag
  scale = np.max(power)
  power /= scale

  parts = []
  if method == 'NMF':
    u, v = NMF(power, k)
    for i in range(k):
      power = (u[:, i:i + 1] @ v[i:i + 1, :]) * scale
      w = (np.sqrt(power)) * angle
      parts.append(ISTFT(w, win, step))
    power = (u @ v) * scale
    w = (np.sqrt(power)) * angle
    parts.append(ISTFT(w, win, step))

  path, ext = os.path.splitext(input_file_name)
  names = [path + str(i) + ext for i in range(k)] + [path + 'all' + ext]
  for output_file_name, part in zip(names, parts):
    write(output_file_name, fs, part)

def get_args():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--file', required=True, help='input wav file')
  parser.add_argument('-k', '--k', required=True, help='#(cluster)')
  parser.add_argument('-m', '--method', required=True, help='from (NMF, K-means, )')
  return vars(parser.parse_args())

def STFT(x, win, step):
  l = len(x)
  N = len(win)
  M = int(ceil(float(l - N + step) / step))

  new_x = zeros(N + ((M - 1) * step), dtype=float64)
  new_x[: l] = x

  w = zeros((M, N), dtype=complex64)
  for m in range(M):
    start = step * m
    w[m, :] = fft(new_x[start : start + N] * win)
  return w

def ISTFT(w, win, step):
  M, N = w.shape
  if (len(win) != N):
    import sys
    sys.exit(1)

  l = (M - 1) * step + N
  x = zeros(l, dtype=float64)
  wsum = zeros(l, dtype=float64)
  for m in range(M):
    start = step * m
    x[start : start + N] = x[start : start + N] + ifft(w[m, :]).real * win
    wsum[start : start + N] += win ** 2
  pos = (wsum != 0)
  x_pre = x.copy()
  x[pos] /= wsum[pos]
  return x.astype(np.int16)

def NMF(x, k, s=0):
  np.random.seed(s)
  m, n = x.shape
  u = np.random.rand(m, k)
  v = np.random.rand(k, n)

#  eps = 0.0001
#  loss = lambda x, u, v: (x * np.log(x / (u @ v)) - x + (u @ v)).sum()

  t = 0
  while t < 40:
    t += 1
    u = u * ((x / ((u @ v) + 1e-15)) @ (v.T / np.sum(v.T, axis=0)))
    v = v * ((u / np.sum(u, axis=0)).T @ (x / ((u @ v) + 1e-15)))

  return u, v

if __name__ == '__main__':
  main()
