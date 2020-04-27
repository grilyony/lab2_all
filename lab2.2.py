import random
import math
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
import time
import cmath
# import pickle
# import os.path
# from googleapiclient.discovery import build
# from google_auth_oauthlib.flow import InstalledAppFlow
# from google.auth.transport.requests import Request
# from apiclient.http import MediaFileUpload


def generate_values(n, N, W):
    generated_signal = np.zeros(N)
    start = time.time()
    for i in range(n):
        fi = 2 * math.pi * random.random()
        A = 5 * random.random()
        w = W - i * W / (n)

        x = A * np.sin(np.arange(0, N, 1) * w + fi)
        generated_signal += x

    print(f"Execution time (Generation): {time.time() - start}")
    #     upload_to_drive("time.txt")
    return generated_signal


def draw(arr, x_label, y_label, title, legend, ax):
    result, = ax.plot(range(len(arr)), arr, '-', label=legend)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return result


def write_file(content, file_name="time.txt"):
    with open(file_name, "w") as f:
        f.write(content + "\n")


# def upload_to_drive(file_name):
#     # If modifying these scopes, delete the file token.pickle.
#     SCOPES = ['https://www.googleapis.com/auth/drive']
#
#     creds = None
#     if os.path.exists('token.pickle'):
#         with open('token.pickle', 'rb') as token:
#             creds = pickle.load(token)
#     # If there are no (valid) credentials available, let the user log in.
#     if not creds or not creds.valid:
#         if creds and creds.expired and creds.refresh_token:
#             creds.refresh(Request())
#         else:
#             flow = InstalledAppFlow.from_client_secrets_file(
#                 'credentials.json', SCOPES)
#             creds = flow.run_local_server(port=0)
#         # Save the credentials for the next run
#         with open('token.pickle', 'wb') as token:
#             pickle.dump(creds, token)
#
#     service = build('drive', 'v3', credentials=creds)
#     file_metadata = {'name': file_name}
#     media = MediaFileUpload(file_name)
#     file = service.files().create(body=file_metadata,
#                                   media_body=media,
#                                   fields='id').execute()


def discrete_fourier(signal):
    start = time.perf_counter_ns()
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N):
        spectre[p] = np.dot(signal, np.cos(2 * math.pi * p / N * np.linspace(0, N - 1, N))) \
                     - 1j * np.dot(signal, np.sin(2 * math.pi * p / N * np.linspace(0, N - 1, N)))
    # print(f"Execution time (DFT): {time.time() - start}")
    return spectre, time.perf_counter_ns() - start


signal = generate_values(10, 256, 900)
fig, axs = plt.subplots(2, 2)

# Дискретне перетворення Фур'є:

spectr, _ = discrete_fourier(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "DFT", "Amplitude", axs[0, 0])
axs[0, 0].legend(handles=[ampl], loc='upper right')
axs[0, 0].grid()

phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "DFT", "Phase", axs[0, 1])
axs[0, 1].legend(handles=[phase], loc='upper right')
axs[0, 1].grid()


# Швидке перетворення Фур'є:


def fast_fourier(signal):
    start = time.perf_counter_ns()
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N // 2):
        E_m = np.dot(
            signal[0:N:2],
            np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))
        ) - 1j * np.dot(
            signal[0:N:2],
            np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))
        )

        W_p = (np.cos(2 * math.pi * p / N) - 1j * np.sin(2 * math.pi * p / N))

        O_m = np.dot(
            signal[1:N:2],
            np.cos(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))
        ) - 1j * np.dot(
            signal[1:N:2],
            np.sin(2 * math.pi * p / (N / 2) * np.arange(0, N // 2, 1))
        )

        spectre[p] = E_m + W_p * O_m
        spectre[p + N // 2] = E_m - W_p * O_m
    # print(f"Execution time (FFT): {time.time() - start}")
    return spectre, time.perf_counter_ns() - start


spectr, _ = fast_fourier(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "FFT", "Amplitude", axs[1, 0])
axs[1, 0].legend(handles=[ampl], loc='upper right')
axs[1, 0].grid()

phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "FFT", "Phase", axs[1, 1])
axs[1, 1].legend(handles=[phase], loc='upper right')
axs[1, 1].grid()
plt.show()


def fast_fourier_parallel(signal):
    start = time.perf_counter_ns()
    p = Pool(2)
    parts = (signal[::2], signal[1::2])
    N = len(signal)
    even, odd = p.map(fft_part, parts)
    coefs = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([even + coefs[:N // 2] * odd,
                           even + coefs[N // 2:] * odd]), time.perf_counter_ns() - start


def fft_part(signal):
    n = len(signal)
    p = np.arange(n)
    k = p.reshape((n, 1))
    w = np.exp(-2j * np.pi * p * k / n)
    return np.dot(w, signal)


fig, axs = plt.subplots(1, 2)
fig.suptitle("Parallel Fast Fourier Transform", fontsize=16)
spectr, _ = fast_fourier_parallel(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "Polar Spectr", "Amplitude", axs[0])
axs[0].legend(handles=[ampl], loc='upper right')
axs[0].grid()

phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "Polar Spectr", "Phase", axs[1])
axs[1].legend(handles=[phase], loc='upper right')
axs[1].grid()
plt.show()

n = 10
time_dft, time_fft, time_fft_parts = (np.zeros(n), np.zeros(n), np.zeros(n))
for i in range(n):
    for _ in range(3):
        signal = generate_values(6, (i + 1) * 256, 2100)
        time_dft[i] = discrete_fourier(signal)[1]
        time_fft[i] = fast_fourier(signal)[1]
        time_fft_parts[i] = fast_fourier_parallel(signal)[1]

time_dft /= 3
time_fft /= 3
time_fft_parts /= 3
dft_l, = plt.plot(range(256, 256 * n + 1, 256), time_dft, label="dft")
fft_l, = plt.plot(range(256, 256 * n + 1, 256), time_fft, label="fft")
fft_parallel_l, = plt.plot(range(256, 256 * n + 1, 256), time_fft_parts, label="fft_parallel")
plt.legend(handles=[dft_l, fft_l, fft_parallel_l])
plt.xlabel("N")
plt.ylabel("Nanoseconds")
plt.show()
