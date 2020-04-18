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


def draw(arr, x_label, y_label, title, legend, file_name=None):
    result, = plt.plot(range(len(arr)), arr, '-', label=legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    return result


def write_file(content, file_name="time.txt"):
    with open(file_name, "w") as f:
        f.write(content + "\n")


# def to_drive(file_name):
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
#
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

# Cигнал:

plot = draw(signal, "t", "x(t)", "Signal", "X(t)", "blue")
plt.grid()
plt.show()

# Амплітудний спектр:

spectr = discrete_fourier(signal)
polar_spectr = np.array(list(map(lambda x: cmath.polar(x), spectr)))
ampl = draw(polar_spectr[:, 0], "p", "A(p)", "Polar Spectr", "Amplitude")
plt.legend(handles=[ampl], loc='upper right')
plt.grid()
plt.show()

# Фазовий спектр:

phase = draw(polar_spectr[:, 1], "p", "Phi(p)", "Polar Spectr", "Phase")
plt.legend(handles=[phase], loc='upper right')
plt.grid()
plt.show()
