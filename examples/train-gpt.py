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

if __name__ == "__main__":

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
            print(    "step:", step,
                    "\t train loss:", "%.2f" % train_loss.item(), 
                    "\t test loss:",  "%.2f" % test_loss.item()   )
