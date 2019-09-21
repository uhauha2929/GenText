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
from models.cvae import SentenceCVAE

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


class CVAE(object):

    def __init__(self):
        self.hidden_dim = 256
        self.embedding_dim = 200
        self.latent_dim = 128
        self.word_dropout_rate = 0.2
        self.anneal_steps = 500
        self.embedding_dropout_rate = 0.2

        self.batch_size = 128
        self.learning_rate = 5e-3
        self.epochs = 10

        self.serialization_dir = 'checkpoints/cvae/'
        self.vocab_dir = 'vocab/cvae/'


if __name__ == '__main__':
    conf = CVAE()
    data_reader = LMReader(return_classes=True)
    train_dataset = data_reader.read('data/tweets/Tweets.csv')

    if os.path.exists(conf.vocab_dir):
        vocab = Vocabulary.from_files(conf.vocab_dir)
    else:
        vocab = Vocabulary.from_instances(chain(train_dataset))
        vocab.save_to_files(conf.vocab_dir)

    print(vocab.get_vocab_size())

    iterator = BasicIterator(batch_size=conf.batch_size)
    iterator.index_with(vocab)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=conf.embedding_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    vae = SentenceCVAE(word_embeddings, conf.hidden_dim, conf.latent_dim,
                       vocab, DEVICE, conf.word_dropout_rate, conf.anneal_steps,
                       conf.embedding_dropout_rate).to(DEVICE)

    optimizer = torch.optim.Adam(vae.parameters(), lr=conf.learning_rate)

    trainer = Trainer(model=vae,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      num_epochs=conf.epochs,
                      cuda_device=DEVICE.index,
                      serialization_dir=conf.serialization_dir)
    trainer.train()
    negative_text_words = vae.generate('negative', batch_size=5000)['texts']
    positive_text_words = vae.generate('positive', batch_size=5000)['texts']
    neutral_text_words = vae.generate('neutral', batch_size=5000)['texts']

    with open('tweets.txt', 'wt', encoding='utf-8') as f:
        for negative, positive, neutral in zip(negative_text_words,
                                               positive_text_words,
                                               neutral_text_words):
            f.write(' '.join(negative) + '\n')
            f.write(' '.join(positive) + '\n')
            f.write(' '.join(neutral) + '\n')

