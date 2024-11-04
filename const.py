LUKE_BASE = 'studio-ousia/luke-base'
LUKE_LARGE = 'studio-ousia/luke-large'
LUKE_LARGE_TACRED = 'studio-ousia/luke-large-finetuned-tacred'

LRS = {LUKE_BASE: 5e-5, LUKE_LARGE: 3e-5, LUKE_LARGE_TACRED: 3e-5}

MODE_CONTRASTIVE_CANDIDATES = 'contr-cand'
MODE_CONTRASTIVE_CLUSTER = 'contr-clust'
MODE_OFFICIAL = 'official'

DEVICE = 'cuda:1'
DATA_DIR = '/data2/nhanse02/thesis/data'
TRAIN_SAMPLES_FP = 'data/train_annotated.json'
DEV_SAMPLES_FP = 'data/dev.json'

PAD_IDS = {'input_ids': 1, 'entity_ids': 0, 'entity_position_ids': -1, 'attention_mask': 0, 'entity_attention_mask': 0}

MAX_ENCODER_LENGTH = 512 # Max length of the encoder input
MAX_DOC_LENGTH = 1024 # Max length a document overall (in tokens)

MAX_GRAD_NORM = 1.0 # Max gradient norm for clipping
WARMUP_RATIO = 0.06
GRADIENT_ACCUMULATION_STEPS = 1