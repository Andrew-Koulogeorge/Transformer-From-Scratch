"""
Building an autoregressive LLM from scratch using PyTorch! 
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import config 


class AutoRegressiveLanguageModel(nn.Module):

        def __init__(self, vocab_size) -> None:
            super().__init__()

            self.token_embedding_table = nn.Embedding(vocab_size,config.MODEL_DIMENSION)
            self.positional_embedding_table = nn.Embedding(config.BLOCK_SIZE, config.MODEL_DIMENSION) 

            # Mistake I made: need to store all the transformer blocks in a Module List. lets all of the modules be properly registered with the parent class
            
            self.transformer_blocks = nn.ModuleList([TransformerBlock() for _ in range(config.NUM_TRANSFORMER_BLOCKS)])
            self.lm_head = nn.Linear(config.MODEL_DIMENSION, vocab_size)

        def forward(self, idx, targets=None):
            
            # the input for the forward pass starts out as a batch of words and it is here that we gather the input dim
            B, T = idx.shape
        
            token_embedding = self.token_embedding_table(idx) # (B,T,d)
            
            pos_vector = torch.arange(0,config.BLOCK_SIZE)
            pos_vector = pos_vector.to(config.DEVICE)
            positional_embedding = self.positional_embedding_table(pos_vector) # (T,d)

            x = token_embedding + positional_embedding # (B,T,d)

            # apply transformer blocks to all input
            for transformer_block in self.transformer_blocks: x = transformer_block(x) # (B,T,d)

            logits = self.lm_head(x) # (B,T,d) --> (B,T,vocab_size)

            if targets == None:
                loss = None
            else:
                """
                per the pytorch documentation, the inputs to CEL needs to be a tensor of size equal to the number of classes (vocab_size) for a non batched input and (B x vocab_size) for a batched input
                """
                B,T,vocab_size = logits.shape

                # need to reformat the logits so that it works with the pytorch cross entropy function
                logits = logits.view(B*T,vocab_size)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
            
            return logits, loss
        
        def generate(self, idx, max_number_generated):
            """
            Want to generate text using the models forward pass. Need to make sure that as we are sending in the context into the model, we are only sending in
            a context length that is the size of its block size (also know as its context window)

            Lets generate outputs for each of the batch examples at once time!
            """

            for _ in range(max_number_generated):
        
                logits, _ = self(idx[:,-config.BLOCK_SIZE:]) # B x T --> B x T x vocab size. 
                last_token_probabilities = torch.softmax(logits[:,-1,:], dim=1) # B x T x vocab size --> B x vocab_size
                idx_next = torch.multinomial(last_token_probabilities, num_samples=1) # B x vocab_size --> B x 1
                idx = torch.cat((idx, idx_next), dim=1) # B x T --> B x (T+1)

            return idx        

class TransformerBlock(nn.Module):
    """
    Bringing together multihead self attention with masking, layer norm, residual connections, and a feedforward network into a single block
    """
    def __init__(self) -> None:
        super().__init__()
        
        self.multihead = MultiHeadSelfAttention(num_heads=config.NUM_HEADS, head_dim=int(config.MODEL_DIMENSION/config.NUM_HEADS))
        self.layer_norm1 = torch.nn.LayerNorm(config.MODEL_DIMENSION, device=config.DEVICE)
        self.feed_forward = FeedForward()
        self.layer_norm2 = torch.nn.LayerNorm(config.MODEL_DIMENSION, device=config.DEVICE)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self, X):

        X = self.layer_norm1(self.multihead(X) + X)
        X = self.dropout(self.feed_forward(X)) + X
        Y = self.layer_norm2(X)
        
        return Y

class MultiHeadSelfAttention(nn.Module):
    """
    Implementing multiple self attention heads in parellel and then concatenating the results together!
    This is why we wanted to have the head dim as an input so we can decrease the dim of each of the heads. Thus, when we put them together it
    will be the same dim as if we did alone big head
    """

    def __init__(self, num_heads, head_dim):
        """
        Make sure that head_dim * num_heads = d! This makes it so the function's input and outputs are of the same dimentions
        """
        super().__init__()
        self.heads = [SelfAttentionHead(attention_dim=head_dim) for _ in range(num_heads)]
    
    def forward(self, X):
        """
        forward: X (B x T x d) --> n * ys (B x T x d/n) --> Y (B x T x d) 
        apply each of the heads on x and concat them all together on the last dim
        """
        
        Y = torch.cat(tuple(head(X) for head in self.heads), dim=2) 
        return Y


class SelfAttentionHead(nn.Module):
    """ 
    NN Decoder block that computes the self attention of an input X of size B x T x d 
    - B is the batch size (examples that we are processing in parallel)
    - T is the block size (the number of tokens that our model can process at a single time)
    - d is the dimention of the token vector embedding space. Note that we will project these tokens into a different attention dim space
    """
    
    def __init__(self, attention_dim) -> None:
        super().__init__()

        # need to define the structure for the attention layer 
        # linear layers at the beggining go from the embedding dim (d) to the attention_dim. This is the data dependent projection!
        
        self.key = nn.Linear(config.MODEL_DIMENSION, attention_dim, bias=False, device=config.DEVICE)
        self.query = nn.Linear(config.MODEL_DIMENSION, attention_dim, bias=False, device=config.DEVICE)
        self.value = nn.Linear(config.MODEL_DIMENSION, attention_dim, bias=False, device=config.DEVICE)

        # buffers are not apart of the learnable parameters of the model but are apart of the models state! 
        # this single line it what makes the attention head a decoder. We are preventing the model from looking ahead

        self.register_buffer(name="mask", tensor=torch.tril(torch.ones(size=(config.BLOCK_SIZE,config.BLOCK_SIZE), device=config.DEVICE)))

        self.attention_dim = attention_dim

    def forward(self, X):
        """
        forward: X (B,T,d) --> Y (B,T,attention_dim)
        X is (B,T,d)
        - B is batch size
        - T is number of tokens in an example
        - d is the embedding dimention of each of the tokens
        """

        # linearly project the input tokens

        Q = self.query(X) # (B,T,attention_dim)
        K = self.key(X) # (B,T,attention_dim)
        V = self.value(X) # (B,T,attention_dim)

        # constructing the attention affinities in 4 lines of code. 
        #  (1) Matrix Mult 
        #  (2) Divide by the square root of the attention dim to make the variance of each of the dot products 1
        #  (3) Make the affinity matrix upper triangular so none of the input tokens are looking ahead
        #  (4) Apply softmax along rows so the "total attention" a single token pays to all other tokens are non negative and sum=1
                # note --> softmax in the context of attention is not making a probability distribution! Not interpreted as p distro

        attention_weights = Q @ torch.transpose(K, 1, 2) # Q * K^T --> (B,T,T)
        attention_weights = attention_weights / (self.attention_dim ** (1/2)) # (Q * K^T)/ root(d) -->  divide by sqrt(dim_head)
        attention_weights = attention_weights.masked_fill(mask=(self.mask == 0), value=(float("-inf")))
        attention_weights = F.softmax(attention_weights, dim = 2) 

        # use the construced coefficents to make a linear combination of the value vectors 

        Y = attention_weights @ V # (B,T,attention_dim)

        return Y 

class FeedForward(nn.Module):
    """
    Feed forward connections will be applied to each of the token vectors separatley. We have a Batch of token vectors each of dim d
    forward: X (B x T x d) --> (B x T x 4d) --> (B x T x d)
    """ 
    def __init__(self):
        super().__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(config.MODEL_DIMENSION, 4*config.MODEL_DIMENSION),
            nn.ReLU(),
            nn.Linear(4*config.MODEL_DIMENSION, config.MODEL_DIMENSION)
        )
        self.feed_forward = self.feed_forward.to(config.DEVICE)
    
    def forward(self, X):
        return self.feed_forward(X)   

