import argparse

ALL_LABELS = ['CID', 'NONE']

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_path',help='Your Word Vocabulary PATH',default='data/vocab.txt')
parser.add_argument('--poses_path',help='Your part of speech PATH',default='data/all_pos.txt')
parser.add_argument('--relations_path',help='Your relations PATH',default='data/all_depend.txt')
parser.add_argument('--batch_size',help='Batch size',default=2)
parser.add_argument('--num_workers',help='Num Workers',default=2)
parser.add_argument('--embeddings_path',help='Embedding path',default='data/w2v/embeddings_matrix.npz')
parser.add_argument('--embedding_size',help='Embedding size',default=300)
parser.add_argument('--pos_embedding_size',help='Embedding size',default=6)
parser.add_argument('--relation_embedding_size',help='Embedding size',default=6)
parser.add_argument('--position_embedding_size',help='Embedding size',default=50)
parser.add_argument('--n_filters',help='Num of Filters',default=128)
parser.add_argument('--lr',help='Learning Rate',default=0.01)
parser.add_argument('--epoch',help='Num Epoch',default=10)
parser.add_argument('--clip_grad',help='Clipping Gradient',default=5)
parser.add_argument('--drop_out',help='Drop out prob',default=0.2)
parser.add_argument('--filters_size',help='Filters Size ',default='2:3:4:5')
parser.add_argument('--hidden',help='Num Hidden Unit',default=250)
parser.add_argument('--verbose',help='Verbose',default=True)
parser.add_argument('--patient',help='Patient',default=3)
parser.add_argument('--early_stopping',help='Use Early Stopping',default=True)

args = parser.parse_args()
UNK = "$UNK$"
MAX_LENGTH = 100
MAX_LENGTH_TO_FIT = 24
VOCAB_PATH = args.vocab_path
POSES_PATH = args.poses_path
RELATIONS_PATH = args.relations_path
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
EMBEDDINGS_PATH = args.embeddings_path
EMBEDIDNG_SIZE = args.embedding_size
POS_EMBEDDING_SIZE = args.pos_embedding_size
RELATION_EMBEDDING_SIZE = args.relation_embedding_size
POSITION_EMBEDDING_SIZE = args.position_embedding_size
N_FILTERS = args.n_filters
LR = args.lr
EPOCH = args.epoch
CLIP_GRAD = args.clip_grad
DROP_OUT = args.drop_out
FILTERS_SIZE = [int(x) for x in args.filters_size.split(":")]
HIDDEN = args.hidden
VERBOSE = args.verbose
PATIENT = args.patient
EARLY_STOPPING = args.early_stopping
