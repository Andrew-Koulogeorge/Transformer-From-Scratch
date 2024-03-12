import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import config 
from gpt import AutoRegressiveLanguageModel
# Tokenizer class to convert the text dataset into sequences of integers that the model can understand

class Tokenizer:

    def __init__(self, vocabulary) -> None:
        """
        want to store the dicts where we can look up charachters <--> integers
        """

        self.char_to_int = {char:integer for integer, char in enumerate(vocabulary)}
        self.int_to_char = {integer:char for integer, char in enumerate(vocabulary)}

    def encode(self, text):
        """
        Function that will take text as input and output a sequence of integers
        """

        integer_sequence = [self.char_to_int[text[i]] for i in range(len(text))]
        return integer_sequence

    def decode(self, integer_seq):
        """
        Function that maps list of integers back to charachters
        """

        words = [self.int_to_char[i] for i in integer_seq]
        return "".join(words)

# load the dataset

def load_dataset():
    
    # open the dataset
    with open("/home/andrew/andrew/Transformer-From-Scratch/tiny-shakespeare.txt", "r") as file:
        data = file.read()

    # tokens of our vocabulary will be on the charachter level
    vocab = sorted(list(set(data)))
    tokenizer = Tokenizer(vocab)

    # encode the dataset with the tokenizer into a sequence of integers and place it into a torch tensor
    tokenized_dataset = torch.tensor(tokenizer.encode(data), dtype=torch.long)

    # 90 10 train test split
    n = int(.9 * len(tokenized_dataset))
    train_split = tokenized_dataset[:n]
    test_split = tokenized_dataset[n:]

    return tokenizer, train_split, test_split


def get_batch(dataset, batch_size):
    """
    During training, want to be able to fetch batches of examples from the training split!
    """

    # randomly generate indexs in the dataset to sample examples
    random_indexs = torch.randint(low=0, high=len(dataset)-config.BLOCK_SIZE, size=(batch_size,))

    # label for the LM task is the next word! Just shifting over the tokens by 1 so that the examples and labels line up
    X = torch.stack([dataset[idx:idx+config.BLOCK_SIZE] for idx in random_indexs], dim=0)
    Y = torch.stack([dataset[idx+1:idx+config.BLOCK_SIZE+1] for idx in random_indexs], dim=0)

    return X, Y

def save_dataset():
    tokenizer, train_set, test_set = load_dataset()
    torch.save(tokenizer, f="tokenizer.pt")
    torch.save(train_set, f="tokenized_train.pt")
    torch.save(test_set, f="tokenized_test.pt")

def train():

    # load in tokenizer and the datasets
    tokenizer = torch.load(f="tokenizer.pt")
    train_split = torch.load(f="tokenized_train.pt")
    test_split = torch.load(f="tokenized_test.pt")


    # define the model and pass its parameters to an optimizer
    model = AutoRegressiveLanguageModel()
    model = model.to(config.DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    loss_tracker = []

    # define a training loop where we get a random batch from the dataset, compute a forward pass and the backwards loss, step, ect

    for step in range(config.ITERATIONS):

        # get a random batch from the dataset
        X, Y = get_batch(train_split, config.BATCH_SIZE)
        X = X.to(config.DEVICE)
        Y = Y.to(config.DEVICE)
        assert X.shape == (config.BATCH_SIZE, config.BLOCK_SIZE), "Input data matrix not correct dimention"

        # zero out the gradient so we correctly compute the gradient (autograd accumulates grads)
        optimizer.zero_grad()

        # compute the forward pass for the model

        logits, loss = model(idx=X, targets=Y)
        assert logits.shape == (config.BATCH_SIZE*config.BLOCK_SIZE, config.VOCAB_SIZE), "Forward pass logits matrix not correct dim"

        # keep track of the loss at each iteration
        loss_tracker.append(loss.item())

        # compute the backwards pass for the gradient of the model with respect to the weights
        loss.backward()

        # step the optimizer to change the models weights
        optimizer.step()

        if step % 500 == 0:
            print(loss.item())
    
    torch.save(obj=model.state_dict(), f="trained_model_thetas.pt")
    
    print(f"Starting loss: {loss_tracker[0]}")
    print(f"Ending loss: {loss_tracker[-1]}")



if __name__ == "__main__":
   
    save_dataset()
    print(f"We saved the datasets sucsessfully!")
    
    train()
    print(f"Finished training the model! ")
   

