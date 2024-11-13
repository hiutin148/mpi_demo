import numpy as np

def generate_and_save_matrices(filename_A, filename_B, n):
    # Tạo ma trận A và B ngẫu nhiên kích thước n x n
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    # Lưu ma trận A và B vào các tệp .npy
    np.save(filename_A, A)
    np.save(filename_B, B)
    print(f"Ma trận A và B đã được lưu vào {filename_A} và {filename_B}")

# Sử dụng n = 5000 cho ví dụ này
n = 5000
generate_and_save_matrices("matrix_A.npy", "matrix_B.npy", n)