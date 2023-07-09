import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # number of parallel independent sequences to run
chunk_size = 8 # max size of the chunks to run algo on
max_iters = 5000 # max times to run algo
eval_interval = 300
learning_rate = 1e-3
# ability to run on gpu if the machine has it (much faster)
device ='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32
url = 'http://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

# get data from Shakespeare text
urllib.request.urlretrieve(url, filename="input.txt")
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# sort all unique characters into a list
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create simple encryption from integer to character
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

# train and test splits
# 80% as training data
# 20% as testing data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.8*len(data))
train_data = data[:n]
val_data = data[n:]

# load the data
# inputs are x, targets are y
def get_batch(split):

    # designates which data to look at
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - chunk_size, (batch_size,))

    # stack a bunch of torch rows on top to get a tensor matrix
    x = torch.stack([data[i:i+chunk_size] for i in ix])
    y = torch.stack([data[i+1:i+chunk_size+1] for i in ix])
    return x, y

# used to perform validation and blocks leaks from test model
# disables gradients temporarily
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Self-attention model
class Head(nn.Module):
    # single head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(chunk_size, chunk_size)))
    
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        # key represents what the node has
        k = self.key(x)
        # query represents what the node searches for / has an affinity for
        q = self.query(x)
        # this line will do the dot product between the key of one node and the query of another
        # this dot product results in the affinity of the two nodes
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        # removes the upper triangular values (to hide the answers)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # softmax fixes the distribution to be sum up to 1 per row
        wei = F.softmax(wei, dim=-1)
        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # keep track of position so that tokens can interact
        self.position_embedding_table = nn.Embedding(chunk_size, n_embed)
        # creating self attention head
        self.sa_head = Head(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # add token and positional embeds
        # get logits based off of that
        token_embed = self.token_embedding_table(idx)
        positional_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embed + positional_embed
        x = self.sa_head(x)
        logits = self.lm_head(x)
        
        # converting B, T, C into B, C, T:
        # loss is the penalty for making a bad guess
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last chunk size token of each row
            # so that the chunk size doesn't overflow
            idx_cond = idx[:, -chunk_size:]
            # get predictions
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            # apply softmax to get probabilities - converts vector into vector of possibilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
# defining model to use
model = BigramLanguageModel()
m = model.to(device)

# using Adam python optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # evaluate the loss on train and val sets once in a while
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))