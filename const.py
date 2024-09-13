LUKE_BASE = 'studio-ousia/luke-base'
LUKE_LARGE = 'studio-ousia/luke-large'
LUKE_LARGE_TACRED = 'studio-ousia/luke-large-finetuned-tacred'

MAX_ENCODER_LENGTH = 512 # Max lenghth of the encoder input
MAX_DOC_LENGTH = 1024 # Max length a document overall (in tokens)

PAD_IDS = {'input_ids': 1, 'entity_ids': 0, 'entity_position_ids': -1, 'attention_mask': 0, 'entity_attention_mask': 0}
