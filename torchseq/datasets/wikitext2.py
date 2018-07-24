import os
import os.path
import torch.utils.data as data
from .utils import download_url
from ..vocab import Dictionary

__all__ = ['WikiText2']

class WikiText2(data.Dataset):

    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    filename = "wikitext-2-v1.zip"
    ufiledir = "wikitext-2"

    base_dir = "wikitext-2"

    vocab_file = "vocab.pkl"

    splits = {
        "train": "wiki.train.tokens",
        "val": "wiki.valid.tokens",
        "test": "wiki.test.tokens"
    }

    def __init__(self, root, split="train", 
            start_token=None, unk_token="<unk>", end_token=None, transform=None, download=False):
        super(WikiText2, self).__init__()
        self.root = os.path.expanduser(root)
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.transform = transform

        self.ufiledir = os.path.join(self.root, self.ufiledir)
        self.base_dir = os.path.join(self.root, self.base_dir)
        self.vocab_file = os.path.join(self.base_dir, self.vocab_file)

        if split not in self.splits:
            raise ValueError("split should be one of {0}, but got {1}".format(
                ",".join(self.splits.keys()), split
                ))

        self.split = os.path.join(self.base_dir, self.splits[split])
        self.data_file = self.split + ".pkl"

        if download:
            self.download()
        else:
            if not os.path.exists(self.base_dir):
                raise RuntimeError("Data not found, use download=True")

        if not os.path.exists(self.data_file):
            print("Processing dataset {0}".format(self.split))
            self.data = self.loadData()
            import pickle
            pickle.dump(self.data, open(self.data_file, "wb"))
        else:
            print("Load processed dataset {0}".format(self.split))
            import pickle
            self.data = pickle.load(open(self.data_file, "rb"))

        if not os.path.exists(self.vocab_file):
            if split != "train":
                raise ValueError("Switch train split to process dataset")
            print("Building vocab")
            self.vocab = self.buildVocab()
            old = self.vocab.trim(min_freq=1)
            print("trim #tokens {0} -> {1} with min_freq={2}".format(old, len(self.vocab), 1))
            self.vocab.saveToFile(self.vocab_file)
        else:
            print("Load built vocab")
            self.vocab = Dictionary()
            self.vocab.loadFromFile(self.vocab_file)

        self.vocabInfo()

    def loadData(self):
        data = []
        with open(self.split, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                
                tokens = line.split(" ")
                if tokens[0] == "=" and tokens[-1] == "=":
                    continue
                
                tokens = [t.lower() for t in tokens]

                if self.start_token is not None:
                    tokens.insert(0, self.start_token)
                if self.end_token is not None:
                    tokens.append(self.end_token)
                
                data.append(tokens)
        return data

    def buildVocab(self):
        dic = Dictionary()

        # FIXME: use n > 1 to prevent from triming
        if self.start_token is not None:
            dic.updateToken(self.start_token, n=2)
        if self.end_token is not None:
            dic.updateToken(self.end_token, n=2)
        if self.unk_token is not None:
            dic.updateToken(self.unk_token, n=2)

        for seq in self.data:
            for token in seq:
                dic.updateToken(token)

        return dic

    def vocabInfo(self):
        print("Vocab Info:")
        print("\t#tokens : {0}".format(len(self.vocab)))

    def download(self):
        import zipfile
        download_url(self.url, self.root, self.filename, md5=None)

        if os.path.exists(self.ufiledir):
            print("Using extracted files at {0}".format(self.ufiledir))
        else:
            print("Extracting at {0}...".format(self.root))
            z = zipfile.ZipFile(os.path.join(self.root, self.filename), "r")
            z.extractall(path=self.root)
            z.close()

    def __getitem__(self, index):
        data = [self.vocab.getTokenId(token, default=self.unk_token) for token in self.data[index][:-1]]
        target = [self.vocab.getTokenId(token, default=self.unk_token) for token in self.data[index][1:]]

        if self.transform is not None:
            data = self.transform(data)
            target = self.transform(target)

        assert(len(data) == len(target))
        return data, target

    def __len__(self):
        return len(self.data)