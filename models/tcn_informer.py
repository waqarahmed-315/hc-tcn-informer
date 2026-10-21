import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tcn import TemporalConvNet
from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.revin import RevIN
from models.initialization import apply_initialization

class TCNInformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=256, n_heads=8, e_layers=2, d_layers=2,
                 d_ff=1024, dropout=0.15, attn="full", embed="timeF", freq="h",
                 activation="gelu", output_attention=False, distil=True, mix=True,
                 device=torch.device("cuda:0"), init_mode="standard",
                 tcn_channels=None, tcn_kernel_size=5, tcn_dropout=0.15,
                 fusion_type="gate", use_revin=True, init_cache="./init_cache",
                 init_config_path=None):
        super().__init__(); self.pred_len=out_len; self.output_attention=output_attention
        self.fusion_type=fusion_type; self.use_revin=use_revin
        if use_revin:
            self.revin_enc=RevIN(enc_in); self.revin_dec=RevIN(c_out)
        if tcn_channels is None: tcn_channels=[128,256,512]
        self.tcn=TemporalConvNet(enc_in,tcn_channels,kernel_size=tcn_kernel_size,dropout=tcn_dropout)
        self.tcn_proj=nn.Linear(tcn_channels[-1],d_model) if tcn_channels[-1]!=d_model else nn.Identity()
        self.tcn_norm=nn.LayerNorm(d_model)
        self.enc_embedding=DataEmbedding(enc_in,d_model,embed,freq,dropout)
        Attn=ProbAttention if attn=="prob" else FullAttention
        self.encoder=Encoder([
            EncoderLayer(AttentionLayer(Attn(False,factor,attention_dropout=dropout,output_attention=output_attention),d_model,n_heads,mix=False),
                         d_model,d_ff,dropout=dropout,activation=activation) for _ in range(e_layers)],
            [ConvLayer(d_model) for _ in range(e_layers-1)] if distil else None, norm_layer=nn.LayerNorm(d_model))
        self.enc_norm=nn.LayerNorm(d_model)
        if fusion_type=="gate":
            self.gate=nn.Linear(d_model*2,d_model); self.sigmoid=nn.Sigmoid()
        elif fusion_type=="concat": self.fusion=nn.Linear(d_model*2,d_model)
        elif fusion_type in {"add","mean"}: self.fusion=nn.Identity()
        else: raise ValueError(f"Unknown fusion_type {fusion_type}")
        self.dec_embedding=DataEmbedding(dec_in,d_model,embed,freq,dropout)
        self.decoder=Decoder([
            DecoderLayer(AttentionLayer(Attn(True,factor,attention_dropout=dropout,output_attention=False),d_model,n_heads,mix=mix),
                         AttentionLayer(FullAttention(False,factor,attention_dropout=dropout,output_attention=False),d_model,n_heads,mix=False),
                         d_model,d_ff,dropout=dropout,activation=activation) for _ in range(d_layers)], norm_layer=nn.LayerNorm(d_model))
        self.projection=nn.Linear(d_model,c_out,bias=True)
        self.initialization_metadata=apply_initialization(
            self, mode=init_mode, config_path=init_config_path, cache_dir=init_cache
        )

    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec,enc_self_mask=None,dec_self_mask=None,dec_enc_mask=None):
        if self.use_revin:
            x_enc=self.revin_enc(x_enc,mode="norm")
            self.revin_dec.mean=self.revin_enc.mean[:,:,-1:]; self.revin_dec.std=self.revin_enc.std[:,:,-1:]
            x_dec=(x_dec-self.revin_dec.mean)/self.revin_dec.std
        tcn_out=self.tcn(x_enc.transpose(1,2)).transpose(1,2)
        tcn_out=self.tcn_norm(self.tcn_proj(tcn_out))
        enc_out=self.enc_embedding(x_enc,x_mark_enc); enc_out,attns=self.encoder(enc_out,attn_mask=enc_self_mask); enc_out=self.enc_norm(enc_out)
        if tcn_out.shape[1]!=enc_out.shape[1]:
            tcn_out=F.adaptive_avg_pool1d(tcn_out.transpose(1,2),enc_out.shape[1]).transpose(1,2)
        if self.fusion_type=="gate":
            gate=self.sigmoid(self.gate(torch.cat([tcn_out,enc_out],dim=-1))); fused=gate*tcn_out+(1-gate)*enc_out
        elif self.fusion_type=="concat": fused=self.fusion(torch.cat([tcn_out,enc_out],dim=-1))
        elif self.fusion_type=="add": fused=tcn_out+enc_out
        else: fused=0.5*(tcn_out+enc_out)
        dec_out=self.dec_embedding(x_dec,x_mark_dec)
        dec_out=self.decoder(dec_out,fused,x_mask=dec_self_mask,cross_mask=dec_enc_mask)
        dec_out=self.projection(dec_out)
        if self.use_revin: dec_out=self.revin_dec(dec_out,mode="denorm")
        out=dec_out[:,-self.pred_len:,:]
        return (out,attns) if self.output_attention else out
