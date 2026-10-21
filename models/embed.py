import torch
import torch.nn as nn
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=padding, padding_mode="circular")
        nn.init.kaiming_normal_(self.tokenConv.weight, mode="fan_in", nonlinearity="leaky_relu")
    def forward(self, x): return self.tokenConv(x.permute(0,2,1)).transpose(1,2)

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0)/d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)
    def forward(self, x): return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super().__init__()
        Embed = FixedEmbedding if embed_type=="fixed" else nn.Embedding
        if freq == "t": self.minute_embed = Embed(4, d_model)
        self.hour_embed    = Embed(24, d_model)
        self.weekday_embed = Embed(7, d_model)
        self.day_embed     = Embed(32, d_model)
        self.month_embed   = Embed(13, d_model)
    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self,"minute_embed") else 0.
        return self.hour_embed(x[:,:,3]) + self.weekday_embed(x[:,:,2]) +                self.day_embed(x[:,:,1]) + self.month_embed(x[:,:,0]) + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super().__init__()
        freq_map = {"h":4,"t":5,"s":6,"m":1,"a":1,"w":2,"d":3,"b":3}
        self.embed = nn.Linear(freq_map[freq], d_model)
    def forward(self, x): return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.value_embedding    = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = (TemporalEmbedding(d_model, embed_type, freq)
                                   if embed_type != "timeF"
                                   else TimeFeatureEmbedding(d_model, embed_type, freq))
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, x_mark):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x) +
                            self.temporal_embedding(x_mark))
