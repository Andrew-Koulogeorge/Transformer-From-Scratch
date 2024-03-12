import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import config 
from gpt import AutoRegressiveLanguageModel
import argparse
# Tokenizer class to convert the text dataset into sequences of integers that the model can understand

class Tokenizer:

    def __init__(self, vocabulary) -> None:
        """
        want to store the dicts where we can look up charachters <--> integers
        """

        self.char_to_int = {char:integer for integer, char in enumerate(vocabulary)}
        self.int_to_char = {integer:char for integer, char in enumerate(vocabulary)}
        self.vocab_size = len(vocabulary)
        print(f"This dataset has a size of :{self.vocab_size}")
        # print(self.char_to_int.keys())

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

def load_dataset(path):
    
    # open the dataset
    with open(path, "r") as file:
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

def save_dataset(dataset, path):
    tokenizer, train_set, test_set = load_dataset(path)
    torch.save(tokenizer, f=f"tokenizer_{dataset}.pt")
    torch.save(train_set, f=f"tokenized_train_{dataset}.pt")
    torch.save(test_set, f=f"tokenized_test_{dataset}.pt")


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

def train():

    # load in tokenizer and the datasets
    tokenizer = torch.load(f="tokenizer_Harry Potter.pt")
    train_split = torch.load(f="tokenized_train_Harry Potter.pt")
    test_split = torch.load(f="tokenized_test_Harry Potter.pt")
    vocab_size = tokenizer.vocab_size

    # define the model and pass its parameters to an optimizer
    model = AutoRegressiveLanguageModel(vocab_size)
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
        assert logits.shape == (config.BATCH_SIZE*config.BLOCK_SIZE, vocab_size), "Forward pass logits matrix not correct dim"

        # keep track of the loss at each iteration
        loss_tracker.append(loss.item())

        # compute the backwards pass for the gradient of the model with respect to the weights
        loss.backward()

        # step the optimizer to change the models weights
        optimizer.step()

        if step % 500 == 0:
            print(f"Iteration {step}/{config.ITERATIONS}: Loss= {loss.item()}")
            
    torch.save(obj=model.state_dict(), f="trained_model_thetas.pt")
    
    print(f"Starting loss: {loss_tracker[0]}")
    print(f"Ending loss: {loss_tracker[-1]}")


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="Harry Potter")
parser.add_argument("--path", type=str, default="/Users/andrewkoulogeorge/Desktop/Dartmouth/Senior/Winterim24/PyTorch/GPT/Harry_Potter_all_books_preprocessed.txt")
parser.add_argument("--download_data", action="store_true")
parser.add_argument("--train", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_path = args.path
    train_loop = args.train
    download = args.download_data
    
    if download:
        save_dataset(dataset_name, dataset_path)
        print(f"We saved the datasets sucsessfully!")   
    
    if train_loop:
        train()
        print(f"Finished training the model! ")
   

