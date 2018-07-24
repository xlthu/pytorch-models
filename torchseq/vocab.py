
class IDFreq():
    def __init__(self, id, freq):
        self.id = id
        self.freq = freq
    
    def increament(self, n):
        self.freq += n

class Dictionary():
    NULL_TOKEN = ""
    NULL_TOKEN_ID = 0

    def __init__(self):
        self.token2id_f = {} # token -> IDFreq
        self.tokens = []

        self.token2id_f[self.NULL_TOKEN] = IDFreq(0, 10000)
        self.tokens.append(self.NULL_TOKEN)
    
    def updateToken(self, token, n=1):
        idf = self.token2id_f.get(token, None)
        if idf is None:
            self.token2id_f[token] = IDFreq(len(self.tokens), n)
            self.tokens.append(token)
        else:
            idf.increament(n)

    def getTokenId(self, token, default=None):
        item = self.token2id_f.get(token, None)
        if item is None:
            return self.token2id_f[default].id
        else:
            return item.id

    def saveToFile(self, fname):
        assert(len(self.token2id_f) == len(self.tokens))
        import pickle
        pickle.dump([self.token2id_f, self.tokens], open(fname, "wb"))
    
    def loadFromFile(self, fname):
        import pickle
        self.token2id_f, self.tokens = pickle.load(open(fname, "rb"))
        assert(len(self.token2id_f) == len(self.tokens))

    def trim(self, min_freq=1):
        old_n_tokens = len(self.tokens)
        self.tokens = [token for token in self.tokens if self.token2id_f[token].freq > min_freq]

        _token2id_f = {}
        for i, token in enumerate(self.tokens):
            _token2id_f[token] = IDFreq(i, self.token2id_f[token].freq)

        self.token2id_f = _token2id_f

        return old_n_tokens

    def __len__(self):
        return len(self.tokens)