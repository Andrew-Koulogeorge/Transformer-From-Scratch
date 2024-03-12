# MODEL HYPER PARAMETERS #
VOCAB_SIZE = 65 # this is only because we are using the char level tokenization from the datset
BLOCK_SIZE = 32
MODEL_DIMENSION = 512
NUM_HEADS = 8
NUM_TRANSFORMER_BLOCKS = 6

# Training Hyperparameters #
LEARNING_RATE = 0.0001
BATCH_SIZE = 16
ITERATIONS = 1000000
DEVICE = "cuda:0"
    