import numpy as np
from feature_engineering.word_embedding.constant import *

WORD_EMBEDDING_FILE = "data/w2v_model/wikipedia-pubmed-and-PMC-w2v.bin"
word_map = None
rel_lut = {}
tag_lut = {}


def __load_word_map():
    global word_map
    print("Load NLPLab WE")
    word_map = {}

    with open(WORD_EMBEDDING_FILE, 'rb') as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size

        vocab_size = 10000
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word)
                    break
                if ch != b'\n':
                    word.append(ch)
            word = word.decode("utf-8")
            word_map[word] = np.fromstring(f.read(binary_len), dtype='float32')
    print("Load finished")


# load word_map once
__load_word_map()


def load_rel_lut(method):
    if method == RELATION_2_VEC_METHOD_ONE_HOT:
        # Initialize one hot relation embedding
        rel_map = {}
        for rel in REL_POS:
            rel_map[rel] = list(np.zeros(len(REL_POS), dtype=float))
            rel_map[rel][REL_POS[rel]] = 1.0
        rel_lut[method] = rel_lut
    elif method == RELATION_2_VEC_METHOD_GLOROT_NORMAL:
        rel_lut[method] = __get_normal_distribution_relation_embedding_map(
            size=REL_DEFAULT_EMBEDDING_SIZE,
            seed=GLOROT_RANDOM_SEED,
            mean=GLOROT_NORMAL_MEAN,
            sigma=GLOROT_NORMAL_SIGMA
        )
    elif method == RELATION_2_VEC_METHOD_GLOROT_UNIFORM:
        rel_lut[method] = __get_uniform_distribution_relation_embedding_map(
            size=REL_DEFAULT_EMBEDDING_SIZE,
            seed=GLOROT_RANDOM_SEED,
            lim=GLOROT_UNIFORM_LIM
        )


def load_tag_lut(method):
    if method == TAG_2_VEC_METHOD_ONE_HOT:
        # Initialize one hot relation embedding
        rel_map = {}
        for tag in TAG_POS:
            rel_map[tag] = list(np.zeros(len(TAG_POS), dtype=float))
            rel_map[tag][TAG_POS[tag]] = 1.0
        tag_lut[method] = rel_lut
    elif method == TAG_2_VEC_METHOD_GLOROT_NORMAL:
        tag_lut[method] = __get_normal_distribution_tag_embedding_map(
            size=REL_DEFAULT_EMBEDDING_SIZE,
            seed=GLOROT_RANDOM_SEED,
            mean=GLOROT_NORMAL_MEAN,
            sigma=GLOROT_NORMAL_SIGMA
        )
    elif method == TAG_2_VEC_METHOD_GLOROT_UNIFORM:
        tag_lut[method] = __get_uniform_distribution_tag_embedding_map(
            size=REL_DEFAULT_EMBEDDING_SIZE,
            seed=GLOROT_RANDOM_SEED,
            lim=GLOROT_UNIFORM_LIM
        )


def __get_uniform_distribution_relation_embedding_map(size=1, seed=0, lim=1.0):
    # Initialize random relation embedding
    np.random.seed(seed)
    ret = {}
    for r in REL_POS:
        # generate size random float with uniform distribution [-lim, lim)
        ret[r] = list(np.random.uniform(-lim, lim, size))

    return ret


def __get_normal_distribution_relation_embedding_map(size=1, seed=0, mean=0.0, sigma=1.0):
    # Initialize random relation embedding
    np.random.seed(seed)
    ret = {}
    for r in REL_POS:
        # 200 random float with uniform distribution [-lim, lim)
        ret[r] = list(sigma * np.random.randn(size) + mean)

    return ret


def __get_uniform_distribution_tag_embedding_map(size=1, seed=0, lim=1.0):
    # Initialize random tag embedding
    np.random.seed(seed)
    ret = {}
    for r in TAG_POS:
        # generate size random float with uniform distribution [-lim, lim)
        ret[r] = list(np.random.uniform(-lim, lim, size))

    return ret


def __get_normal_distribution_tag_embedding_map(size=1, seed=0, mean=0.0, sigma=1.0):
    # Initialize random tag embedding
    np.random.seed(seed)
    ret = {}
    for r in TAG_POS:
        # 200 random float with uniform distribution [-lim, lim)
        ret[r] = list(sigma * np.random.randn(size) + mean)

    return ret


class NLPLab:
    def __init__(self):
        pass

    @staticmethod
    def sdp2matrix(sdp, word_options, rel_options, tag_options=None, remove_ends=False):
        """
        :param remove_ends:
        :param str sdp:
        :param dict word_options:
        :param dict rel_options:
        :param dict tag_options:
        :return: np.array
        """
        embed_size = word_options.get(WORD_OPTION_SIZE_KEY, 0)
        if tag_options is not None:
            embed_size += tag_options.get(TAG_OPTION_SIZE_KEY, 0)

        path = sdp.split() if not remove_ends else sdp.split()[1:-1]
        if len(path) > DEFAULT_SENTENCE_LENGTH:
            started = (len(path) - DEFAULT_SENTENCE_LENGTH) // 2
            path = path[started:started + DEFAULT_SENTENCE_LENGTH]

        ret = np.zeros((DEFAULT_SENTENCE_LENGTH, embed_size))

        for i in range(len(path)):
            if path[i][0] == '(' and path[i][-1] == ')':
                # ret[i] = np.zeros(embed_size)
                ret[i] = NLPLab.rel2vec(
                    path[i],
                    method=rel_options.get(REL_OPTION_METHOD_KEY, RELATION_2_VEC_METHOD_ONE_HOT),
                    size=embed_size
                )
            else:
                if tag_options is None:
                    ret[i] = NLPLab.word2vec(path[i])
                else:
                    word, tag = path[i].rsplit('/', 1)
                    vec = list(NLPLab.word2vec(
                        word,
                        size=word_options.get(WORD_OPTION_SIZE_KEY, 0)
                    )) + list(NLPLab.tag2vec(
                        tag,
                        method=tag_options.get(TAG_OPTION_METHOD_KEY, TAG_2_VEC_METHOD_ONE_HOT),
                        size=tag_options.get(TAG_OPTION_SIZE_KEY, 0)
                    ))
                    ret[i] = vec

        return ret

    @staticmethod
    def word2vec(word, size=WORD_DEFAULT_EMBEDDING_SIZE):
        if word in word_map:
            w = word_map[word]
            return NLPLab.padding(w, size)
        else:
            return np.zeros(size)

    @staticmethod
    def rel2vec(rel, method=RELATION_2_VEC_METHOD_ONE_HOT, size=REL_DEFAULT_EMBEDDING_SIZE):
        # return np.zeros(size)
        # Check for punctuation
        if rel == '-PUNC-':
            return np.array([10 for _ in range(size)])

        if method not in rel_lut:
            load_rel_lut(method)

        lut = rel_lut.get(method)

        # Standardize the relation
        rel = rel.lower()
        rel = '(' + rel.rsplit('_', 1)[-1]
        # s_rel = rel.split(':')[0] + ')' if rel not in REL_POS and ':' in rel else rel
        s_rel = rel.split(':')[0] + ')' if ':' in rel else rel

        if s_rel in lut:
            w = lut[s_rel]
            return NLPLab.padding(w, size)
        else:
            return np.zeros(size)

    @staticmethod
    def tag2vec(tag, method=TAG_2_VEC_METHOD_ONE_HOT, size=TAG_DEFAULT_EMBEDDING_SIZE):
        if method not in tag_lut:
            load_tag_lut(method)

        lut = tag_lut.get(method)

        if tag in lut:
            w = lut[tag]
            return NLPLab.padding(w, size)
        else:
            return np.zeros(size)

    @staticmethod
    def padding(array, size):
        if len(array) == size:
            return np.array(array)
        elif len(array) > size:
            return np.array(array[:size])
        else:
            r = np.zeros(size)
            r[:len(array)] = array
            return r
