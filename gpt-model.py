import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F


batch_size = 64 # number of parallel independent sequences to run
chunk_size = 256 # max size of the chunks to run algo on
max_iters = 5000 # max times to run algo
eval_interval = 500
learning_rate = 3e-4
# ability to run on gpu if the machine has it (much faster)
device ='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
# dropout drops certain amounts of nodes during testing to provide more distinct sub NNs
dropout = 0.2
# get data from internet
url = 'INSERT URL HERE'


# set data as a input text file so we can call on it later
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
    x, y = x.to(device), y.to(device)
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

        self.dropout = nn.Dropout(dropout)
    
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

        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out


# implementing multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # create x many heads
        # run them in parallel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # proj transforms the layer to become linear
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # concatenate the heads' outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# adding feed forward mechanism
class FeedForward(nn.Module):
    # creating a linear layer followed by a non-linear one
    # this allows for the model to account for non-linear results
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            # edited the inner layer of feed forward to be multiplied by 4 in channel size
            # this allows for better non-linear results to match the residual layer addition
            nn.Linear(n_embed, 4 * n_embed),
            # rectified linear activation unit
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


# adding a transformer block
class Block(nn.Module):
    # allows for multiple multihead self-attentions to get called
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.feedforward = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # implementing residual layer
        # residual layer allows for layers input to use the output from a previous layer
        # implementing layer norm before each step (self-attention and feed forward)
        x = x + self.sa(self.ln1(x))
        x = x + self.feedforward(self.ln2(x))
        return x


# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # keep track of position so that tokens can interact
        self.position_embedding_table = nn.Embedding(chunk_size, n_embed)
        
        # creating transformer block _ times with n_layer heads
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        # final layer norm
        self.ln_final = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # add token and positional embeds
        # get logits based off of that
        token_embed = self.token_embedding_table(idx)
        positional_embed = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embed + positional_embed
        # calling the transformer block
        x = self.blocks(x)
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
