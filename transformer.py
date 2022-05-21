from audioop import bias
import torch 
import torch.nn as nn 


# We are going to start with the self attn mechansim
class SelfAttention(nn.Module):
    # Recall from paper, we have an embed size of 256. We are
    # going to split the embeddings into 8 heads of 32 each
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # Lets make an assert message to ensure we can evenely spilt embed size
        assert(self.head_dim*heads ==embed_size), "The embedding size has to be envenly divisible by number of heads"

        # Now we can proceed to define the linear layers that we're sending our Q,V,K vals through
        # Figure 2 of paper
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Then we add the fully connected linear layer that occurs after. dot product and concatenation
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # We are going to get number of training examples
        # The number of queries we are sending in 
        N = query.shape[0]
        # should correspond to source and target sentence length
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #Split embedding into self.heads pieces. 
        # Split embeddings into multiple heads
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum()
        # queries shape: (N, query_len, heads, heads_dim)
        # key shape: (N, key_len, heads, heads_dim)
        # energy shape: (N,  heads, query_len, key_len)

