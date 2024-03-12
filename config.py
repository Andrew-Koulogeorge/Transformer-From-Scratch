# MODEL HYPER PARAMETERS #

BLOCK_SIZE = 16 # context window of model
MODEL_DIMENSION = 512 # latent dim
NUM_HEADS = 8 # number of heads in multiheaded attention
NUM_TRANSFORMER_BLOCKS = 1 # stack of transformer blocks

# Training Hyperparameters 

LEARNING_RATE = 0.0001
BATCH_SIZE = 8
ITERATIONS = 100000
DEVICE = "cpu"
    