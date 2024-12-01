import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_attention_heads):
        super(SelfAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads #12
        self.query = nn.Linear(embedding_dim, hidden_size)
        self.key = nn.Linear(embedding_dim, hidden_size)
        self.value = nn.Linear(embedding_dim, hidden_size)
        self.output = nn.Linear(embedding_dim, hidden_size)

    def forward(self,x):
        batch_size = x.size(0)
        seq_length = x.size(1)

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        query = query.view(batch_size,seq_length,self.num_attention_heads,-1)
        key = key.view(batch_size,seq_length,self.num_attention_heads,-1)
        value = value.view(batch_size,seq_length,self.num_attention_heads,-1)

        query =  query.permute(0,2,1,3)
        key =  key.permute(0,2,1,3)
        value =  value.permute(0,2,1,3)

        attention_scores = torch.matmul(query,key.permute(0,1,3,2))/math.sqrt(self.embedding_dim/self.num_attention_heads)
        attention_scores = F.softmax(attention_scores, dim=-1)

        value = torch.matmul(attention_scores,value)

        output = value.permute(0,2,1,3)
        output = output.reshape(batch_size,seq_length,-1)

        output = self.output(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, embedding_dim):
        super(FeedForward, self).__init__()
        self.input = nn.Linear(embedding_dim, 4*embedding_dim)
        self.output = nn.Linear( 4*embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = self.output(x)
        return x

class Transformer(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_attention_heads, num_layers, dropout_rate):
        super(Transformer, self).__init__()
        self.attention = SelfAttention(embedding_dim, hidden_size, num_attention_heads)
        self.ff = FeedForward(embedding_dim)
        self.LayerNorm1 = nn.LayerNorm(embedding_dim)
        self.LayerNorm2 = nn.LayerNorm(embedding_dim)
    def forward(self, x):
        for i in range(12):
            x = self.attention(x)
            x = x + x
            x = self.LayerNorm1(x)
            x = self.ff(x)
            x = x + x
            x = self.LayerNorm2(x)
        return x

class Bert(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_attention_heads, num_layers, dropout_rate):
        super(Bert, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=embedding_dim, embedding_dim=embedding_dim)
        self.position_embedding = nn.Embedding(num_embeddings=embedding_dim, embedding_dim=embedding_dim)
        self.token_embedding = nn.Embedding(num_embeddings=embedding_dim, embedding_dim=embedding_dim)
        self.LayerNorm = nn.LayerNorm(embedding_dim)
        self.transformer = Transformer(embedding_dim, hidden_size, num_attention_heads, num_layers, dropout_rate)
        self.output = nn.Linear(embedding_dim, 30522)

    def forward(self, x):
        x1 = self.embedding(x)
        x2 = self.position_embedding(x)
        x3 = self.token_embedding(x)
        x = x1 + x2 + x3
        x = self.LayerNorm(x)
        x = self.transformer(x)
        x = self.output(x)
        return x