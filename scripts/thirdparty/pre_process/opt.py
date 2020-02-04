from scripts.thirdparty.pre_process.segmenters.simple import SimpleSegmenter
from scripts.thirdparty.pre_process.segmenters.nltk import NltkSegmenter
from scripts.thirdparty.pre_process.segmenters.spacy import SpacySegmenter
from scripts.thirdparty.pre_process.segmenters.corenlp import CorenlpSegmenter

from scripts.thirdparty.pre_process.tokenizers.simple import SimpleTokenizer
from scripts.thirdparty.pre_process.tokenizers.spacy import SpacyTokenizer
from scripts.thirdparty.pre_process.tokenizers.corenlp import CorenlpTokenizer

from scripts.thirdparty.pre_process.options.normalizers import NumericNormalizer
from scripts.thirdparty.pre_process.options.snowball_stemmer import SnowballStemmer
from scripts.thirdparty.pre_process.options.wordnet_lemmatizer import WordNetLemmatizer


# Options for segmenters
SimpleSegmenter = SimpleSegmenter
NltkSegmenter = NltkSegmenter
SpacySegmenter = SpacySegmenter
CorenlpSegmenter = CorenlpSegmenter

# Options for tokenizers
SimpleTokenizer = SimpleTokenizer
SpacyTokenizer = SpacyTokenizer
CorenlpTokenizer = CorenlpTokenizer

# Options for optional processes
NumericNormalizer = NumericNormalizer
SnowballStemmer = SnowballStemmer
WordNetLemmatizer = WordNetLemmatizer

# Keys
TOKENIZER_KEY = "tokenizer"
SEGMENTER_KEY = "segmenter"
OPTION_KEY = "options"
