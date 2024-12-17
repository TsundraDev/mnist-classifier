import struct
import numpy as np


def read_file_img(file):
    with open(file, "rb") as f:
        magic, size, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"{file}: wrong magic number")

        data = bytearray(f.read())
        images = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
        images = images.reshape([size, rows * cols])
    images /= 256
    return images

def read_file_label(file):
    with open(file, "rb") as f:
        magic, size = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"{file}: wrong magic number")

        labels = np.zeros([size, 10])
        for i in range(size):
            index = int.from_bytes(f.read(1))
            labels[i][index] = 1
    return labels
