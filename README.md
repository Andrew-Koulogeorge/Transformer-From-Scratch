# Transformer from Scratch
This codebase contains an Autoregressive Language Model trained on the Harry Potter books. The Language Model is powered by six decoder Transformer blocks, which I implemented from scratch in PyTorch. The model was trained with a context window of 64 tokens, which is very small, considering each token in the model's vocabulary is a single character. The hidden dimension of the model is 512, and each multiheaded self-attention block has eight heads. The model was trained with Adam with a learning rate of 1e-4 and batch size of 32 over 100,000 gradient update steps.

Here is a sample of the model's output. The model was trained on the language modeling objective function, and it appears to have learned a thing or two about JK Rowling's writing with only about an hour of training:

"""
Work in progress: Add model generation given test set prompt of Harry Potter text.
"""

The model's performance is still very limited. The transformer learned how to construct words and phrases correctly from Harry Potter, but its lack of context length, naive decoding strategy (I used greedy decoding), and character-level vocabulary are all preventing the model from constructing fluent sentences. This project is ongoing, and I hope to continue improving the model's output by incorporating Byte Pair Encoding for more complex tokenization and longer training time.

