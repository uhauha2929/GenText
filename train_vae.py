# -*- coding: utf-8 -*-
# @Time    : 2019/9/18 12:06
# @Author  : uhauha2929
import os
from itertools import chain
import torch
from allennlp.data import Vocabulary
from allennlp.data.iterators import BasicIterator
from allennlp.modules import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training import Trainer

from dataset.readers import LMReader
from models.vae import SentenceVAE

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


class VAE(object):

    def __init__(self):
        self.hidden_dim = 256
        self.embedding_dim = 200
        self.latent_dim = 128
        self.word_dropout_rate = 0.2
        self.num_anneal = 300
        self.embedding_dropout_rate = 0.2

        self.batch_size = 128
        self.learning_rate = 8e-3
        self.epochs = 10

        self.serialization_dir = 'checkpoints/vae/'
        self.vocab_dir = 'vocab/vae/'


if __name__ == '__main__':
    conf = VAE()

    data_reader = LMReader()
    train_dataset = data_reader.read('data/tweets/Tweets.csv')

    if os.path.exists(conf.vocab_dir):
        vocab = Vocabulary.from_files(conf.vocab_dir)
    else:
        vocab = Vocabulary.from_instances(chain(train_dataset))
        vocab.save_to_files(conf.vocab_dir)

    print(vocab.get_vocab_size())

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=conf.embedding_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    vae = SentenceVAE(word_embeddings, conf.hidden_dim, conf.latent_dim,
                      vocab, DEVICE, conf.word_dropout_rate, conf.num_anneal,
                      conf.embedding_dropout_rate).to(DEVICE)

    iterator = BasicIterator(batch_size=conf.batch_size)
    iterator.index_with(vocab)

    optimizer = torch.optim.Adam(vae.parameters(), lr=conf.learning_rate)

    trainer = Trainer(model=vae,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      num_epochs=conf.epochs,
                      cuda_device=DEVICE.index,
                      serialization_dir=conf.serialization_dir)
    trainer.train()
    text_words = vae.generate(batch_size=20)['texts']
    for words in text_words:
        print(' '.join(words))
