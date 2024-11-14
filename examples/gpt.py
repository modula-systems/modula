import time

import torch
import numpy as np

# Karpathy's smallest GPT config

vocab_size = 65
context = 64
num_heads = 4
d_embed = 128
d_query = 32
d_value = 32
num_blocks = 4

# training hparams

init_lr = 0.5
wd = 0.01
batch_size = 12
steps = 2001
eval_steps = 100
log_interval = 200

chars = list("\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    global stoi
    return [stoi[c] for c in s]

def decode(l):
    global itos
    return ''.join([itos[i] for i in l])

# let's start by defining our GPT architecture
# (we could instead just import GPT from modula.compound)

from modula.atom import *
from modula.bond import *

def Attention(num_heads, d_embed, d_query, d_value, context, causal):
    """Multi-head attention."""
    Q = AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    K = AddHeads(num_heads) @ Linear(num_heads * d_query, d_embed)
    V = AddHeads(num_heads) @ Linear(num_heads * d_value, d_embed)
    W = Linear(d_embed, d_value * num_heads) @ RemoveHeads()

    return W @ FunctionalAttention(causal) * 1/3 @ (Q, K, V)

def GPT(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks, blocks_mass=5):
    """GPT."""
    token_embedding = Embedding(vocab_size, d_embed)
    position_embedding = Embedding(context, d_embed) @ Enumerate()
    initial = 1/2 * token_embedding + 1/2 * position_embedding
    initial.tare()

    attention = Attention(num_heads, d_embed, d_query, d_value, context, causal=True) @ LayerNorm()
    mlp = Linear(d_embed, 4*d_embed) @ ScaledGELU() @ Linear(4*d_embed, d_embed) @ LayerNorm()
    attention_block = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * attention
    mlp_block       = (1-1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    blocks = (mlp_block @ attention_block) ** num_blocks
    blocks.tare(absolute=blocks_mass)

    final = Linear(vocab_size, d_embed) @ LayerNorm()

    return final @ blocks @ initial

# now let's set up some data loading utils

class RandomSampler(torch.utils.data.Sampler):

    def __init__(self, data, batch_size):
        self.length = len(data)
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield np.random.randint(self.length, size=self.batch_size)

class SimpleLLMDataset(torch.utils.data.Dataset):

    def __init__(self, data, context):
        self.data = data
        self.context = context

    def __getitem__(self, index):
        return torch.tensor(self.data[index  :index+self.context  ].astype(np.int64)), \
               torch.tensor(self.data[index+1:index+self.context+1].astype(np.int64))

    def __len__(self):
        return len(self.data) - self.context - 1

# now let's start doing stuff

def train():
    # load the data

    trainset = SimpleLLMDataset(np.memmap("examples/data/shakespeare/train.bin", dtype=np.uint16, mode='r'), context)
    testset  = SimpleLLMDataset(np.memmap("examples/data/shakespeare/val.bin",   dtype=np.uint16, mode='r'), context)

    train_sampler = RandomSampler(trainset, batch_size)
    test_sampler  = RandomSampler(testset,  batch_size)

    train_loader = torch.utils.data.DataLoader( trainset, num_workers=1, pin_memory=True, batch_sampler=train_sampler)
    test_loader  = torch.utils.data.DataLoader( testset,  num_workers=1, pin_memory=True, batch_sampler=test_sampler)

    train_iterator = iter(train_loader)
    test_iterator  = iter(test_loader)

    getBatch = lambda train: next(train_iterator if train else test_iterator)

    # load the model

    gpt = GPT(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks)
    weights = gpt.initialize(device="cpu")

    # initialize the Adam state

    beta1 = 0.9
    beta2 = 0.99

    with torch.no_grad():
        mom1 = 0 * weights
        mom2 = 0 * weights

    # train the model

    start = time.time()
    for step in range(steps):

        if step % log_interval == 0:
            test_loss = test_acc = 0
            for eval_step in range(eval_steps):
                data, target = getBatch(train = False)
                output = gpt.forward(data, weights)
                output = output.view(-1, output.size(-1))
                target = target.view(-1)

                with torch.no_grad():
                    test_acc += (output.argmax(dim=1) == target).sum() / target.numel()
                    error = - output[range(target.shape[0]),target] + output.logsumexp(dim=1)
                    test_loss += error.mean()
            test_loss /= eval_steps
            test_acc /= eval_steps

        data, target = getBatch(train = True)
        output = gpt.forward(data, weights)
        output = output.view(-1, output.size(-1))
        target = target.view(-1)

        train_acc = (output.argmax(dim=1) == target).sum() / target.numel()
        error = - output[range(target.shape[0]),target] + output.logsumexp(dim=1)
        train_loss = error.mean()

        train_loss.backward()

        with torch.no_grad():
            grad = weights.grad()

            # adam logic
            mom1 += (1-beta1)**(step/(step+1)) * (grad    - mom1)
            mom2 += (1-beta2)**(step/(step+1)) * (grad**2 - mom2)
            update = mom1 / mom2 ** 0.5
            update.zero_nans()

            schedule = 1 - step / steps

            # modular normalization and weight update
            gpt.normalize(update, target_norm = init_lr * schedule)
            weights -= update
            gpt.regularize(weights, strength = init_lr * schedule * wd)
            weights.zero_grad()

        if step % log_interval == 0:
            print(     "step:", step,
                    "\t train loss:", "%.2f" % train_loss.item(),
                    "\t test loss:",  "%.2f" % test_loss.item()   ,
                   f"\t took: {time.time() - start:.2f}s")
            start = time.time()

    return weights


def inference(weights, input_text, chars_to_generate):
    gpt = GPT(vocab_size, context, num_heads, d_embed, d_query, d_value, num_blocks)
    print(input_text, end="", flush=True)
    context_tokens = torch.tensor(encode(input_text)).unsqueeze(0)
    for _ in range(chars_to_generate):
        with torch.no_grad():
            output = gpt.forward(context_tokens, weights)
        logits = output[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        print(decode([next_token]), end="", flush=True)
        context_tokens = torch.cat([context_tokens, torch.tensor([[next_token]])], dim=1)
        if context_tokens.shape[1] > context:
            context_tokens = context_tokens[:, -context:]


if __name__ == "__main__":
    import argparse

    default_weights_filename = "examples/data/shakespeare/weights.pt"
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--weights", "-w", default=default_weights_filename, help="Weights filename")

    parser_inference = subparsers.add_parser("inference")
    parser_inference.add_argument("--weights", "-w", default=default_weights_filename, help="Weights filename")
    parser_inference.add_argument("--chars", "-c", default=1024)
    parser_inference.add_argument("input", help="Text to be feed into the model")

    args = parser.parse_args()

    if args.mode == "train":
        weights_filename = args.weights

        weights = train()
        torch.save(weights, weights_filename)
        print(f"Weights saved to {weights_filename}")

    elif args.mode == "inference":
        weights_filename = args.weights
        input_text = args.input
        chars_to_generate = args.chars

        print(f"Loading weights from {weights_filename}")
        weights = torch.load(weights_filename)
        print()
        inference(weights, input_text, chars_to_generate)
