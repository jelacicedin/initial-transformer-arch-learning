import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 32
# --------------------------------------------------------------

with open("/code/dataset/tiny-shakespeare.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Sort the chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenize input text and create mapping from chars to ints
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    itos[i] for i in l
)  # decoder: take a list of integers, output a string

# Tokenize Shakespeare
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    """
    Generate a small batch of data of inputs x and targets y
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


# Loss estimate
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)

            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Implement the bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C), torch.arange gives ints [0,...,T-1]
        x = (
            token_emb + pos_emb
        )  # (B, T, C) # x holds not just the token identities but also the positions at which they occur, not overly useful rn since it's all just a bigram model
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(
                B * T, C
            )  # preserve the channel dim, stretch out hte array
            targets = targets.view(B * T)  # one-dimensional
            loss = F.cross_entropy(logits, targets)

        return (
            logits,
            loss,  # logits are the SCORES for the next character in the sequence!!!!!!
        )
        # Therefore you're actually returning a distribution (that we will presumably softmax)

    def generate(self, idx, max_new_tokens):
        """
        Function kept fixed for all models.
        """
        idx = idx.to(device)
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the preds
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomse (B,C)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1)  # (B,C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1).to(device)  # (B,1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1).to(device)  # (B, T+1)
        return idx


model = BigramLanguageModel().to(device)

# create an optimizer and train the model
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train_loss {losses['train']:4f} val loss {losses['val']:4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
