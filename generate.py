import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import config 
from gpt import AutoRegressiveLanguageModel
from train import Tokenizer
import argparse

def generate_text(prompt, num_tokens_to_generate):
    """
    Given a starting prompt, generate text from the LLM
    """

    tokenizer = torch.load(f="tokenizer_HarryPotter.pt")

    print(tokenizer.vocab)

    model_trained = AutoRegressiveLanguageModel(vocab_size=tokenizer.vocab_size)
    trained_model_state = torch.load("trained_model_thetas_harry_potter.pt", map_location="cpu")
    model_trained.load_state_dict(state_dict=trained_model_state, )
    
    model_random = AutoRegressiveLanguageModel(vocab_size=tokenizer.vocab_size)

    encoded_prompt = tokenizer.encode(prompt)
    
    idx = torch.unsqueeze(torch.tensor(encoded_prompt), dim=0)

    model_trained = model_trained.to("cpu")
    model_trained = model_trained.to("cpu")

    model_trained.eval()
    model_random.eval()

    output1 = torch.squeeze(model_random.generate(idx, max_number_generated=num_tokens_to_generate)).tolist()
    output2 = torch.squeeze(model_trained.generate(idx, max_number_generated=num_tokens_to_generate)).tolist()
    
    print(f"This is the random model: {tokenizer.decode(output1)}\n\n")
    print(f"This is the trained model: {tokenizer.decode(output2)} ")

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--length", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()
    prompt = args.prompt
    length = args.length

    generate_text(prompt=prompt, num_tokens_to_generate=length)
