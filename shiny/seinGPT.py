import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

block_size = 128
n_embd = 1024
device = 'cuda' if torch.cuda.is_available() else 'cpu'

enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

#tokens assumed to be encoded
def generateOutput(model, tokens):
    tokens = torch.tensor(tokens)[None,:].to(device)
    result = model.generate(tokens)[0]
    #return decodeTokens(result.tolist())
    return result

def MakeSeinfeldModel(path):
    model = LanguageModel(vocab_size, n_embd).to(device)
    model.load_state_dict(torch.load(path))
    return model

def encodeTokens(tokens):
    return enc.encode(tokens)
    #return torch.Tensor(enc.encode(tokens))[None, :].to(device)

def decodeTokens(tokens):
    return enc.decode(tokens.tolist())

class LanguageBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = AttentionMultiHead(n_head, head_size, n_embd)
        self.ffwd = FeedForwardBlock(n_embd)
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.embedding_size = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        numBlocks = 6
        blocks = []
        for _ in range(numBlocks):
            blocks.append(LanguageBlock(n_embd, 16))
        blocks.append(nn.LayerNorm(n_embd))
        self.blocks = nn.Sequential(*blocks)
        
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    
    def forward(self, idx, targets=None):
        idx = idx.to(device)[:,-block_size:]
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T).to(device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits  = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            targets = targets.to(device)
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx):
        for _ in range(2):
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
class AttentionHead(nn.Module):
    def __init__(self, head_size, n_embd):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(n_embd,n_embd)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v   = self.value(x)
        out = wei @ v
        return out

class AttentionMultiHead(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size, n_embd) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForwardBlock(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
        )
    def forward(self, x):
        out = self.net(x)
        return out