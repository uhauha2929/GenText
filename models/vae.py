# -*- coding: utf-8 -*-
# @Time    : 2019/9/18 9:21
# @Author  : uhauha2929
from typing import Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Vocabulary
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states, sequence_cross_entropy_with_logits
from allennlp.training.metrics import Metric
from overrides import overrides


class SentenceVAE(Model):

    def __init__(self, embedder: TextFieldEmbedder,
                 hidden_dim: int,
                 latent_dim: int,
                 vocab: Vocabulary,
                 device: torch.device,
                 word_dropout_rate: float = 0.2,
                 num_anneal: int = 500,
                 embedding_dropout_rate: float = 0.0):
        super().__init__(vocab)

        self.embedder = embedder
        self.embedding_dim = embedder.get_output_dim()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.vocab = vocab
        self.device = device
        self.word_dropout_rate = word_dropout_rate
        self.num_anneal = num_anneal
        self.embedding_dropout = nn.Dropout(embedding_dropout_rate)

        self.encoder_rnn = PytorchSeq2SeqWrapper(
            torch.nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True))

        self.decoder_rnn = PytorchSeq2SeqWrapper(
            torch.nn.GRU(self.embedding_dim, self.hidden_dim, num_layers=1, batch_first=True))

        self.hidden2mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.hidden2log_var = nn.Linear(self.hidden_dim, self.latent_dim)
        self.latent2hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.outputs2vocab = nn.Linear(self.hidden_dim, self.vocab.get_vocab_size())

        self.metrics = {}
        self.step = 0

    @staticmethod
    def kl_anneal_function(step, num_anneal: int = 300):
        return min(1, step / num_anneal)

    def forward(self, input_tokens: Dict[str, torch.Tensor],
                output_tokens: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(input_tokens)
        batch_size, num_tokens = mask.size()

        embeddings = self.embedder(input_tokens)
        encoder_rnn_outputs = self.encoder_rnn(embeddings, mask)
        encoder_final_states = get_final_encoder_states(encoder_rnn_outputs, mask)

        z_mean = self.hidden2mean(encoder_final_states)
        z_log_var = self.hidden2log_var(encoder_final_states)

        # sampling
        epsilon = torch.randn([batch_size, self.latent_dim], device=self.device)
        z = z_mean + torch.exp(z_log_var / 2) * epsilon

        # decoder
        hidden = torch.tanh(self.latent2hidden(z)).unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand((batch_size, num_tokens), device=self.device)
            prob[mask == 0] = 1  # exclude pad
            input_tokens['tokens'] = input_tokens['tokens'].clone()
            input_tokens['tokens'][prob < self.word_dropout_rate] = \
                self.vocab.get_token_index(DEFAULT_OOV_TOKEN)

        decoder_input_embeddings = self.embedder(input_tokens)
        decoder_input_embeddings = self.embedding_dropout(decoder_input_embeddings)
        decoder_rnn_outputs = self.decoder_rnn(decoder_input_embeddings, mask=mask, hidden_state=hidden)

        # project outputs to vocab
        logits = self.outputs2vocab(decoder_rnn_outputs)
        ent_loss = sequence_cross_entropy_with_logits(logits, output_tokens['tokens'], mask)
        kl_loss = - 0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        self.step += 1
        loss = ent_loss + (self.kl_anneal_function(self.step, self.num_anneal)) * kl_loss

        self.metrics['ent_loss'] = ent_loss.item()
        self.metrics['kl_loss'] = kl_loss.item()

        return {'loss': loss}

    def generate(self, batch_size: int = 1, max_len: int = 50):
        z = torch.randn([batch_size, self.latent_dim]).to(self.device)
        hidden = self.latent2hidden(z).unsqueeze(0)

        current_token_indexes = torch.full((batch_size, 1), self.vocab.get_token_index(START_SYMBOL),
                                           dtype=torch.long, device=self.device)
        generated_texts = []
        end_flags = [False] * batch_size
        for i in range(max_len):
            current_embeddings = self.embedder({"tokens": current_token_indexes})
            current_decoder_rnn_outputs, hidden = self.decoder_rnn._module(current_embeddings, hidden)

            current_logits = self.outputs2vocab(current_decoder_rnn_outputs)
            current_probs = torch.log_softmax(current_logits.squeeze(1), dim=-1)
            current_token_indexes = torch.argmax(current_probs, dim=-1, keepdim=True)

            for j, index in enumerate(current_token_indexes.flatten()):
                if end_flags[j]:
                    continue
                token = self.vocab.get_token_from_index(index.item())
                if token == END_SYMBOL:
                    end_flags[j] = True
                else:
                    if len(generated_texts) - 1 < j:
                        generated_texts.append([])
                    generated_texts[j].append(token)

        return {'texts': generated_texts}

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        updated_metrics = {}
        for metric_name, metric in self.metrics.items():
            if isinstance(metric, Metric):
                updated_metrics[metric_name] = metric.get_metric(reset)
            else:
                updated_metrics[metric_name] = metric
        return updated_metrics
