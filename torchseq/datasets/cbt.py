import os
import os.path
import torch.utils.data as data
from .utils import download_url
from ..vocab import Dictionary

__all__=['CBT']

class CBT(data.Dataset):

    url = "http://www.thespermwhale.com/jaseweston/babi/CBTest.tgz"
    filename = "CBTest.tgz"
    ufiledir = "CBTest"

    base_dir = "CBTest/data"

    vocab_file = "vocab.pkl"

    word_types = ["CN", "NE", "P", "V"]

    splits = {
        "train": "cbtest_{word_type}_train.txt",
        "val": "cbtest_{word_type}_valid_2000ex.txt",
        "test": "cbtest_{word_type}_2500ex.txt"
    }

    def __init__(self, root, word_type="CN", split="train", unk_token="<unk>", transform=None, download=False):
        super(CBT, self).__init__()
        self.root = os.path.expanduser(root)
        self.unk_token = unk_token
        self.transform = transform

        self.ufiledir = os.path.join(self.root, self.ufiledir)
        self.base_dir = os.path.join(self.root, self.base_dir)
        self.vocab_file = os.path.join(self.base_dir, self.vocab_file)

        if word_type not in self.word_types:
            raise ValueError("word_type should be one of {0}, but got {1}".format(
                ",".join(self.word_types), word_type
                ))

        if split not in self.splits:
            raise ValueError("split should be one of {0}, but got {1}".format(
                ",".join(self.splits.keys()), split
                ))
        
        self.word_type = word_type
        self.split = os.path.join(self.base_dir, self.splits[split].format(word_type=self.word_type))
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

            # Uncomment the following line to trim vocab
            old = self.vocab.trim(min_freq=1)
            print("trim #tokens {0} -> {1} with min_freq={2}".format(old, len(self.vocab), 1))
            self.vocab.saveToFile(self.vocab_file)
        else:
            print("Load built vocab")
            self.vocab = Dictionary()
            self.vocab.loadFromFile(self.vocab_file)

        self.vocabInfo()

    def loadData(self):
        def tokenize(line: str):
            return line.strip().split(" ")

        data = []
        documents = []
        with open(self.split, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    documents = []
                    continue

                id, line = line.split(' ', 1)
                line = line.lower()
                if id == "21":
                    # Answer line
                    query, answer, _, candidates = line.split("\t")
                    query = tokenize(query)
                    answer = answer.strip()
                    candidates = candidates.split("|")

                    data.append({
                        "documents": documents,
                        "query": query,
                        "answer": answer,
                        "candidates": candidates
                    })
                else:
                    # Document line
                    documents.extend(tokenize(line))

        return data

    def buildVocab(self):
        dic = Dictionary()

        # FIXME
        if self.unk_token is not None:
            dic.updateToken(self.unk_token, n=2)

        for item in self.data:
            for word in item["documents"]:
                dic.updateToken(word)
            for word in item["query"]:
                dic.updateToken(word)
            dic.updateToken(item["answer"])
            for word in item["candidates"]:
                dic.updateToken(word)

        return dic
 
    def vocabInfo(self):
        print("Vocab Info:")
        print("\t#tokens : {0}".format(len(self.vocab)))

    def download(self):
        import tarfile
        download_url(self.url, self.root, self.filename, md5=None)

        if os.path.exists(self.ufiledir):
            print("Using extracted files at {0}".format(self.ufiledir))
        else:
            print("Extracting at {0}...".format(self.root))
            z = tarfile.open(os.path.join(self.root, self.filename), "r:gz")
            z.extractall(path=self.root, )
            z.close()

    def idify(self, item, default_token):
        return {
            "documents": [self.vocab.getTokenId(token, default=default_token) for token in item["documents"]],
            "query": [self.vocab.getTokenId(token, default=default_token) for token in item["query"]],
            "answer": self.vocab.getTokenId(item["answer"], default=default_token),
            "candidates": [self.vocab.getTokenId(token, default=default_token) for token in item["candidates"]],
        }

    def __getitem__(self, index):
        data = self.idify(self.data[index], default_token=self.unk_token)

        if self.transform is not None:
            for item in data:
                data[item] = self.transform(data[item])

        return data

    def __len__(self):
        return len(self.data)