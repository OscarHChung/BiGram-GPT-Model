import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # number of parallel independent sequences to run
chunk_size = 8 # max size of the chunks to run algo on
max_iters = 3000 # max times to run algo
eval_interval = 300
learning_rate = 1e-2
device ='cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
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

# simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        
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
            # get predictions
            logits, loss = self(idx)
            logits = logits[:, -1, :]

            # apply softmax to get probabilities - converts vector into vector of possibilities
            probs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
# defining model to use
model = BigramLanguageModel(vocab_size)
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