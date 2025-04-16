import numpy as np
import gzip
import os
import struct
import urllib.request

def load_mnist(normalize=True):
    """
    Downloads (if needed) and loads the MNIST dataset.
    Returns: (train_images, train_labels, test_images, test_labels)
    """
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), "mnist_files")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download files if needed
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
             "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.isfile(filepath):
            url = base_url + filename
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, filepath)

    # Load the data
    def parse_images(filepath):
        with gzip.open(filepath, "rb") as f:
            _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols)

    def parse_labels(filepath):
        with gzip.open(filepath, "rb") as f:
            _, num_labels = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    # Load and optionally normalize images
    train_images = parse_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
    test_images = parse_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
    train_labels = parse_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
    test_labels = parse_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))

    if normalize:
        train_images = train_images.astype(np.float32) / 127.5 - 1.
        test_images = test_images.astype(np.float32) / 127.5 - 1.

    return train_images, train_labels, test_images, test_labels