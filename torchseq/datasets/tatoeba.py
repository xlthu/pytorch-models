import random
import os
import os.path
import re
import unicodedata
import torch.utils.data as data
from .utils import download_url
from ..vocab import Dictionary

__all__ = ['Tatoeba']

class Tatoeba(data.Dataset):

    url = "http://www.manythings.org/anki/{lang}-eng.zip"
    filename = "{lang}-eng.zip"
    ufiledir = "{lang}-eng"

    base_dir = "{lang}-eng"

    vocab_lang_file = "vocab-{lang}.pkl"
    vocab_eng_file = "vocab-eng.pkl"

    whole_file = "{lang}.txt"
    splits = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt"
    }

    def __init__(self, root, lang="deu", split="train", 
            start_token=None, unk_token="<unk>", end_token=None, transform=None, download=False):
        super(Tatoeba, self).__init__()
        # start_token only for eng; unk_token, end_token for both
        self.root = os.path.expanduser(root)
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token
        self.transform = transform
        
        # TODO: Support more dataset
        if lang != "deu":
            raise ValueError("Only support German-English Dataset")
        self.lang = lang

        self.url = self.url.format(lang=lang)
        self.filename = self.filename.format(lang=lang)
        self.ufiledir = os.path.join(self.root, self.ufiledir.format(lang=lang))
        self.base_dir = os.path.join(self.root, self.base_dir.format(lang=lang))
        self.vocab_lang_file = os.path.join(self.base_dir, self.vocab_lang_file.format(lang=lang))
        self.vocab_eng_file = os.path.join(self.base_dir, self.vocab_eng_file)
        self.whole_file = os.path.join(self.base_dir, self.whole_file.format(lang=lang))

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

        if not os.path.exists(self.split):
            train, val, test = 0.8, 0.1, 0.1
            print("Split dataset with portion train={train} val={val} test={test}".format(
                train=train, val=val, test=test
            ))
            self.splitDataset(train, val, test)

        if not os.path.exists(self.data_file):
            print("Processing dataset {0}".format(self.split))
            self.data = self.loadData()
            import pickle
            pickle.dump(self.data, open(self.data_file, "wb"))
        else:
            print("Load processed dataset {0}".format(self.split))
            import pickle
            self.data = pickle.load(open(self.data_file, "rb"))

        if not os.path.exists(self.vocab_lang_file) or not os.path.exists(self.vocab_eng_file):
            if split != "train":
                raise ValueError("Switch train split to process dataset")
            print("Building vocab")
            self.vocab_lang, self.vocab_eng = self.buildVocab()
            
            # old = self.vocab_lang.trim(min_freq=1)
            # print("[{lang}] trim #tokens {0} -> {1} with min_freq={2}".format(old, len(self.vocab_lang), 1, lang=lang))
            self.vocab_lang.saveToFile(self.vocab_lang_file)

            # old = self.vocab_eng.trim(min_freq=1)
            # print("[{lang}] trim #tokens {0} -> {1} with min_freq={2}".format(old, len(self.vocab_eng), 1, lang="eng"))
            self.vocab_eng.saveToFile(self.vocab_eng_file)
        else:
            print("Load built vocab")
            self.vocab_lang = Dictionary()
            self.vocab_lang.loadFromFile(self.vocab_lang_file)

            self.vocab_eng = Dictionary()
            self.vocab_eng.loadFromFile(self.vocab_eng_file)

        self.vocabInfo()

    @staticmethod
    def processSentence(s):
        # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
        def unicode2ascii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )

        # Lowercase, trim, and remove non-letter characters
        def normalize(s):
            s = unicode2ascii(s.lower().strip())
            s = re.sub(r"([.!?])", r" \1", s)
            s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
            return s.split()
        
        return normalize(s)

    def loadData(self):
        data = []
        with open(self.split, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                
                pair = [self.processSentence(p) for p in line.split("\t")]
                pair.reverse()

                if self.start_token is not None:
                    pair[1].insert(0, self.start_token)
                if self.end_token is not None:
                    for p in pair:
                        p.append(self.end_token)
                
                data.append(pair)
        return data

    def buildVocab(self):
        dic_lang = Dictionary()
        dic_eng = Dictionary()

        # FIXME: use n > 1 to prevent from triming
        if self.start_token is not None:
            dic_eng.updateToken(self.start_token, n=2)
        if self.end_token is not None:
            dic_lang.updateToken(self.end_token, n=2)
            dic_eng.updateToken(self.end_token, n=2)
        if self.unk_token is not None:
            dic_lang.updateToken(self.unk_token, n=2)
            dic_eng.updateToken(self.unk_token, n=2)

        for lang, eng in self.data:
            for token in lang:
                dic_lang.updateToken(token)
            for token in eng:
                dic_eng.updateToken(token)

        return dic_lang, dic_eng

    def vocabInfo(self):
        print("Vocab Info:")
        print("\t[{lang}]#tokens : {0}".format(len(self.vocab_lang), lang=self.lang))
        print("\t[{lang}]#tokens : {0}".format(len(self.vocab_eng), lang="eng"))

    def download(self):
        import zipfile
        download_url(self.url, self.root, self.filename, md5=None)

        if os.path.exists(self.ufiledir):
            print("Using extracted files at {0}".format(self.ufiledir))
        else:
            print("Extracting at {0}...".format(self.ufiledir))
            z = zipfile.ZipFile(os.path.join(self.root, self.filename), "r")
            if os.mkdir(self.ufiledir):
                raise RuntimeError("Create directory {0} error".format(self.ufiledir))
            z.extractall(path=self.ufiledir)
            z.close()

    def splitDataset(self, train, val, test):
        def dump(l, f):
            with open(f, "w", encoding="utf-8") as fout:
                fout.writelines(l)

        with open(self.whole_file, "r", encoding="utf-8") as f:
            lines = [line for line in f.readlines() if line.strip()]
            n_val = int(len(lines) * val)
            n_test = int(len(lines) * test)
            n_train = len(lines) - n_val - n_test
            
            random.shuffle(lines)

            dump(lines[0:n_val], os.path.join(self.base_dir, self.splits["val"]))
            dump(lines[n_val:(n_val + n_test)], os.path.join(self.base_dir, self.splits["test"]))
            dump(lines[(n_val + n_test):], os.path.join(self.base_dir, self.splits["train"]))

            print("Total lines {0}: Train {1} Val {2} Test {3}".format(len(lines), n_train, n_val, n_test))

    def __getitem__(self, index):
        data = [self.vocab_lang.getTokenId(token, default=self.unk_token) for token in self.data[index][0]]
        target = [self.vocab_eng.getTokenId(token, default=self.unk_token) for token in self.data[index][1]]

        if self.transform is not None:
            data = self.transform(data)
            target = self.transform(target)

        return data, target

    def __len__(self):
        return len(self.data)