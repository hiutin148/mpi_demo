import numpy as np

# Đọc các file kết quả
C_parallel = np.load("C_parallel_4.npy")
C_sequential = np.load("C_sequential.npy")

if np.allclose(C_parallel, C_sequential, atol=1e-8):
    print("Kết quả gần giống nhau!")
else:
    print("Kết quả khác nhau.")
