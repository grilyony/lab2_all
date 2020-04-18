import random
import math
import matplotlib.pyplot as plt
import numpy as np
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
    start = time.time()
    N = len(signal)
    spectre = np.zeros(N, dtype=np.complex64)
    for p in range(N):
        spectre[p] = np.dot(signal, np.cos(2 * math.pi * p / N * np.linspace(0, N - 1, N))) \
                     - 1j * np.dot(signal, np.sin(2 * math.pi * p / N * np.linspace(0, N - 1, N)))
    print(f"Execution time (DFT): {time.time() - start}")
    return spectre


signal = generate_values(10, 256, 900)
fig, axs = plt.subplots(2, 2)

# Дискретне перетворення Фур'є:

spectr = discrete_fourier(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "DFT", "Amplitude", axs[0, 0])
axs[0, 0].legend(handles=[ampl], loc='upper right')
axs[0, 0].grid()

phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "DFT", "Phase", axs[0, 1])
axs[0, 1].legend(handles=[phase], loc='upper right')
axs[0, 1].grid()


# Швидке перетворення Фур'є:


def fast_fourier(signal):
    start = time.time()
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
    print(f"Execution time (FFT): {time.time() - start}")
    return spectre


spectr = fast_fourier(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "FFT", "Amplitude", axs[1, 0])
axs[1, 0].legend(handles=[ampl], loc='upper right')
axs[1, 0].grid()

phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "FFT", "Phase", axs[1, 1])
axs[1, 1].legend(handles=[phase], loc='upper right')
axs[1, 1].grid()
plt.show()
