import numpy as np
import time
from mpi4py import MPI

# Hàm nhân ma trận song song theo khối
def matrix_multiply_parallel_block(A, B, comm, rank, size, block_size):
    n = A.shape[0]
    C = np.zeros((n, n))  # Khởi tạo ma trận kết quả C cho tất cả tiến trình
    
    # Phân chia các khối của ma trận A và B cho các tiến trình
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            if rank == 0:
                # Gửi các khối của ma trận A và B cho các tiến trình khác
                for k in range(1, size):
                    A_block = A[i:i + block_size, k*block_size:(k+1)*block_size]
                    B_block = B[k*block_size:(k+1)*block_size, j:j + block_size]
                    comm.send(A_block, dest=k, tag=11)
                    comm.send(B_block, dest=k, tag=22)
                
                # Nhân khối của tiến trình chủ
                C_block = np.dot(A[i:i + block_size, :block_size], B[:block_size, j:j + block_size])
                C[i:i + block_size, j:j + block_size] += C_block
                
                # Nhận kết quả từ các tiến trình khác và ghép vào C
                for k in range(1, size):
                    C_part = comm.recv(source=k, tag=33)
                    C[i:i + block_size, j:j + block_size] += C_part

            else:
                # Nhận các khối của ma trận A và B từ tiến trình chủ
                A_block = comm.recv(source=0, tag=11)
                B_block = comm.recv(source=0, tag=22)

                # Thực hiện nhân khối của tiến trình này
                C_part = np.dot(A_block, B_block)
                C_part

                # Gửi kết quả về tiến trình chủ
                comm.send(C_part, dest=0, tag=33)
    return C

def load_matrices(filename_A, filename_B):
    # Đọc ma trận A và B từ tệp
    A = np.load(filename_A)
    B = np.load(filename_B)
    return A, B

def main():
    # Đọc ma trận A và B từ tệp
    A, B = load_matrices("matrix_A.npy", "matrix_B.npy")

    # Tính toán song song
    start_time = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    block_size = int(A.shape[0] / size)  # Kích thước khối nhỏ để cải thiện hiệu quả
    C_parallel = matrix_multiply_parallel_block(A, B, comm, rank, size, block_size)
    end_time = time.time()
    if rank == 0:
        print("Thời gian thực hiện song song theo khối: ", end_time - start_time)
        np.save(f"C_parallel_{size}.npy", C_parallel)

if __name__ == "__main__":
    main()

