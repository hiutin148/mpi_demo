import numpy as np
import time
from mpi4py import MPI

# Nhân ma trận tuần tự
def matrix_multiply_seq(A, B):
    return np.dot(A, B)


def load_matrices(filename_A, filename_B):
    # Đọc ma trận A và B từ tệp
    A = np.load(filename_A)
    B = np.load(filename_B)
    return A, B

def main():
    # Đọc ma trận A và B từ tệp
    A, B = load_matrices("matrix_A.npy", "matrix_B.npy")

    # Tính toán tuần tự
    start_time = time.time()
    C_seq = matrix_multiply_seq(A, B)
    end_time = time.time()
    print(f"Thời gian thực hiện tuần tự: {end_time - start_time:.6f} giây")
    np.save("C_sequential.npy", C_seq)
    
if __name__ == "__main__":
    main()
