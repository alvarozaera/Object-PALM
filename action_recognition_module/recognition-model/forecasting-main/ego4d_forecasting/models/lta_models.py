#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Alvaro Zaera de la Fuente

"""Definition of model used for Action Recognition."""


import torch
from torch.distributions.categorical import Categorical
import math
from einops import rearrange
import torch.nn as nn

from .build import MODEL_REGISTRY



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        #self.register_buffer("pe", pe)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))
        
    def forward(self, x):
        x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)



@MODEL_REGISTRY.register()
class ActionRecognitionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
        num_heads = cfg.MODEL.TRANSFORMER_HEADS
        num_layers = cfg.MODEL.TRANSFORMER_LAYERS 
        dim_in = cfg.MODEL.MULTI_INPUT_FEATURES
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim_in, num_heads),
            num_layers,
            norm=nn.LayerNorm(dim_in),
        )
        self.pos_encoder = PositionalEncoding(dim_in, dropout=0.2)

        n_labels_noun = cfg.MODEL.NUM_CLASSES[1]

        bert_embeds_path = cfg.MODEL.BERT_EMBEDS_PATH
        embeds = []
        for noun_idx in range(n_labels_noun):
            filename_bert = f'{bert_embeds_path}/{noun_idx}.pt'
            bert_feature = torch.load(filename_bert, map_location='cpu')
            bert_feature.requires_grad = False
            embeds.append(bert_feature)
        embeds = torch.stack(embeds, dim=0)

        self.label_embeds = nn.Embedding.from_pretrained(embeds, freeze=True)
        
        self.ram_embed_mlp = nn.Sequential(
                                nn.Linear(256, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, dim_in)
                            )
        

        self.label_pos_encoder = PositionalEncoding(dim_in, dropout=0.2)

        self.feature_fusion_egovlp = nn.Linear(12*256, cfg.MODEL.MULTI_INPUT_FEATURES)

        self.verb_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.noun_token = nn.Parameter(torch.randn(1, 1, dim_in))

        self.classifier_dropout = nn.Dropout(cfg.MODEL.DROPOUT_RATE)
        self.verb_classifier = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES[0])
        self.noun_classifier = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES[1])


    def forward(self, x):
        # x is a tuple containing the following elements:
        # - Features EgoVLP: [(B, 12*256) for N_INPUTS input videos] (N_INPUTS=4 BY DEFAULT)
        # - RAM++ Indices: [(B, MAX_NOUNS) for N_INPUTS input videos] (Ego4D taxonomy indices of the nouns found using RAM++)
        # - Padding Mask Nouns: [(B, MAX_NOUNS) for N_INPUTS input videos] (Indices with 1 indicate padding; 0 are actual nouns indices found by RAM++)
        features_egovlp, ram_plus_indices, padding_mask_nouns = x
        chunks = len(features_egovlp) # chunks=N_INPUTS

        # Initialize noun and verb tokens for N_INPUTS actions (assuming cfg.FORECASTING.NUM_INPUT_CLIPS=cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT=N_INPUTS)
        verb_tokens = self.verb_token.repeat(chunks, features_egovlp[0].shape[0], 1) # (N_INPUTS, B, D)
        noun_tokens = self.noun_token.repeat(chunks, features_egovlp[0].shape[0], 1) # (N_INPUTS, B, D)

        # Batch the N_INPUTS elements of the list into a single batched embedding
        features_egovlp = torch.cat(features_egovlp, dim=0) # (N_INPUTS*B, 12*256)
        ram_plus_indices = torch.cat(ram_plus_indices, dim=0) # (N_INPUTS*B, MAX_NOUNS)                    

        # Apply a linear layer to fuse the EgoVLP features of dimension 12*256 into a single D-dimensional feature
        features_egovlp = self.feature_fusion_egovlp(features_egovlp) # (N_INPUTS*B, D)
        
        # Rearrange the EgoVLP features
        features_egovlp = torch.chunk(features_egovlp, chunks=chunks, dim=0) # [(B,D) for N_INPUTS videos]
        features_egovlp = torch.stack(features_egovlp, dim=0) # (N_INPUTS,B,D)

        # Apply positional encoding to the EgoVLP features
        features_egovlp = self.pos_encoder(features_egovlp)
       
        # Getting the DistilBERT embeddings for the top MAX_NOUNS nouns found by RAM++
        features_ram = self.label_embeds(ram_plus_indices) # (N_INPUTS*B, MAX_NOUNS, 256)
        # Apply a mlp to fuse the DistilBERT embeddings of dimension 256 into a single D-dimensional feature
        features_ram = self.ram_embed_mlp(features_ram) # (N_INPUTS*B, MAX_NOUNS, D)

        # Rearrange the features of the nouns found by RAM++
        features_ram = torch.chunk(features_ram, chunks=chunks, dim=0)
        features_ram = torch.stack(features_ram, dim=0) # (N_INPUTS, B, MAX_NOUNS, D)
        features_ram = rearrange(features_ram, 'S B N D -> S (B N) D') # (N_INPUTS, B*MAX_NOUNS, D)
        
        # Apply positional encoding to the features of the nouns found by RAM++ depending on the position of the video in the input list where they were found
        features_ram = self.label_pos_encoder(features_ram)
        features_ram = rearrange(features_ram, 'S (B N) D -> (S N) B D', B=features_egovlp.shape[1]) # (N_INPUTS*MAX_NOUNS, B, D)

        # Rearrange the padding mask
        padding_mask_nouns = torch.stack(padding_mask_nouns, dim=0) # (N_INPUTS, B, MAX_NOUNS)
        padding_mask_nouns = rearrange(padding_mask_nouns, 'S B N -> B (S N)') # (B, N_INPUTS*NOUNS_IDX)


        # Add the noun and verb modality tokens to the EgoVLP features
        noun_tokens = features_egovlp + noun_tokens
        verb_tokens = features_egovlp + verb_tokens

        # Concatenate the noun and verb tokens (two tokens per video to be processed)
        tgt = torch.cat([noun_tokens, verb_tokens], dim=0) # (2*N_INPUTS, B, D)
        # Using the features of the nouns found by RAM++ for cross-attentions
        memory = features_ram

        # Apply the transformer decoder
        x = self.decoder(tgt, memory, memory_key_padding_mask=padding_mask_nouns) # (2*N_INPUTS, B, D)

        # Apply dropout
        x = self.classifier_dropout(x)

        # Classify the verbs and nouns
        noun_preds = self.noun_classifier(x[:chunks]) # (N_INPUTS, B, #nouns)
        verb_preds = self.verb_classifier(x[chunks:]) # (N_INPUTS, B, #verbs)

        # Rearrange the predictions
        noun_preds = noun_preds.transpose(0, 1) # (B, N_INPUTS, #nouns)
        verb_preds = verb_preds.transpose(0, 1) # (B, N_INPUTS, #verbs)

        return [verb_preds, noun_preds]
    

    def generate(self, x, k=1):
        x = self.forward(x)

        results = []
        for head_x in x:
            if head_x.numel() == 0:
                results.append([])
                continue
            if k>1:
                preds_dist = Categorical(logits=head_x)
                preds = [preds_dist.sample() for _ in range(k)]
            elif k==1:
                preds = [head_x.argmax(2)]
            head_x = torch.stack(preds, dim=1)
            results.append(head_x)

        return results