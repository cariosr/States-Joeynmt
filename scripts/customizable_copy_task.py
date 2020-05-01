#!/usr/bin/env python

import numpy as np
import os
import argparse

np.random.seed(42)


def generate_samples(n=10000, low=0, high=10, maxlen=10, letters=True):
    samples = []  
    if letters:
        for i in range(n):
            size = np.random.randint(1, maxlen + 1)
            sample = np.random.randint(low, high, size)
            sample = [chr(ord('a') + x) for x in sample]
            samples.append(sample)
    else:
        for i in range(n):
            size = np.random.randint(1, maxlen + 1)
            sample = np.random.randint(low, high, size)
            samples.append(sample)
    return samples

def sample_to_str(sample):
    return " ".join(map(str, sample))


def save_samples(samples, prefix="train", ext="src", reverse=False):
    with open(prefix + "." + ext, mode="w") as f:
        for sample in samples:
            sample = sample[::-1] if reverse else sample
            f.write(sample_to_str(sample) + "\n")

def generate_task(sample_size=10000, dev_size=500, test_size=500, high=26, maxlen=20, train="train", dev="dev", test="test", src="src", trg="trg", letters=True):

    # train
    samples = generate_samples(sample_size, high=high, maxlen=maxlen, letters=letters)
    save_samples(samples, prefix=train, ext=src, reverse=False)
    save_samples(samples, prefix=train, ext=trg, reverse=False)

    # dev
    samples = generate_samples(dev_size, high=high, maxlen=maxlen, letters=letters)
    save_samples(samples, prefix=dev, ext=src, reverse=False)
    save_samples(samples, prefix=dev, ext=trg, reverse=False)

    # test
    samples = generate_samples(test_size, high=high, maxlen=maxlen, letters=letters)
    save_samples(samples, prefix=test, ext=src, reverse=False)
    save_samples(samples, prefix=test, ext=trg, reverse=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get size and complexity of copy task.')
    for argument in ["sample_size", "dev_size", "test_size", "high", "maxlen"]:
        parser.add_argument(argument, type=int)
    parser.add_argument("letters", type=int, default=int(1))
    args = parser.parse_args()
    generate_task(args.sample_size, args.dev_size, args.test_size, args.high, args.maxlen, letters=bool(args.letters))