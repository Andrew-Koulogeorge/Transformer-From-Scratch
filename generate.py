import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import config 
from gpt import AutoRegressiveLanguageModel
import argparse

def generate_text(prompt, num_tokens_to_generate):
    """
    Given a starting prompt, generate
    """

    tokenizer = torch.load(f="tokenizer.pt")

    model_trained = AutoRegressiveLanguageModel()
    trained_model_state = torch.load("trained_model_thetas.pt")
    model_trained.load_state_dict(state_dict=trained_model_state)
    
    model_random = AutoRegressiveLanguageModel()

    encoded_prompt = tokenizer.encode(prompt)
    
    idx = torch.unsqueeze(torch.tensor(encoded_prompt), dim=0)

    model_trained.eval()
    model_trained.eval()

    output1 = torch.squeeze(model_random.generate(idx, max_number_generated=num_tokens_to_generate)).tolist()
    output2 = torch.squeeze(model_trained.generate(idx, max_number_generated=50)).tolist()
    
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
