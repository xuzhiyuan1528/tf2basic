import numpy as np
from sklearn.model_selection import train_test_split


class ToyDataset:

    def __init__(self, min_len, max_len):
        self.SOS = "<s>"
        self.EOS = "/<s>"

        self.characters = list("abcd")
        self.int2char = self.characters
        # 1 for SOS, 1 for EOS, 1 for padding
        self.char2int = {c: i+3 for i, c in enumerate(self.characters)}

        self.vocab_size = len(self.characters)
        self.min_str_len = min_len
        self.max_str_len = max_len

        self.max_seq_len = max_len + 2

    def get_dataset(self, num_samples):
        inp_set = []
        tar_set = []
        for _ in range(num_samples):
            i, t = self._sample()
            inp_set.append(i)
            tar_set.append(t)
        return inp_set, tar_set

    def split_dataset(self, inp_set, tar_set, test_ratio=0.2):
        return train_test_split(inp_set, tar_set, test_size=test_ratio)

    def _sample(self):
        random_len = np.random.randint(self.min_str_len, self.max_str_len+1)
        random_char = np.random.choice(self.characters, random_len)

        inp = [self.char2int.get(c) for c in random_char]
        tar = inp[::-1]
        inp = [1] + inp + [2]
        tar = [1] + tar + [2]

        inp = np.pad(inp, (0, self.max_str_len+2-len(inp)), 'constant', constant_values='0')
        tar = np.pad(tar, (0, self.max_str_len+2-len(tar)), 'constant', constant_values='0')

        return inp, tar

toy = ToyDataset(5, 10)
inp_set, tar_set = toy.get_dataset(10)
toy.split_dataset(inp_set, tar_set, 0.2)
