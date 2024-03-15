# MODEL HYPER PARAMETERS #

BLOCK_SIZE = 64 # context window of model
MODEL_DIMENSION = 512 # latent dim
NUM_HEADS = 8 # number of heads in multiheaded attention
NUM_TRANSFORMER_BLOCKS = 6 # stack of transformer blocks

# Training Hyperparameters 

LEARNING_RATE = 0.0001
BATCH_SIZE = 32
ITERATIONS = 100000
DEVICE = "cpu" 
    