# -*- coding: utf-8 -*-
# @Time    : 2019/9/18 12:12
# @Author  : uhauha2929
from typing import Dict
import logging
import csv

from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data import Token
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.token_indexers.token_indexer import TokenIndexer
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer

logger = logging.getLogger(__name__)


class LMReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 return_classes: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._output_indexer = {"tokens": SingleIdTokenIndexer()}
        self._return_classes = return_classes

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        instance_strings = []
        classes = []
        with open(file_path, "r") as text_file:
            reader = csv.reader(text_file)
            next(reader)
            for line in reader:
                text = line[10].strip().lower()
                if len(text.split()) > 0:  # remove empty line
                    instance_strings.append(text)
                    classes.append(line[1].strip().lower())

        for s, c in zip(instance_strings, classes):
            yield self.text_to_instance(s, c if self._return_classes else None)

    @overrides
    def text_to_instance(self, sentence: str, class_name: str = None) -> Instance:
        tokenized_string = self._tokenizer.tokenize(sentence)
        tokenized_string.insert(0, Token(START_SYMBOL))
        tokenized_string.append(Token(END_SYMBOL))
        input_field = TextField(tokenized_string[:-1], self._token_indexers)
        output_field = TextField(tokenized_string[1:], self._output_indexer)

        labels = {'input_tokens': input_field, 'output_tokens': output_field}
        if class_name is not None:
            class_field = LabelField(class_name, label_namespace='class_labels')
            labels['classes'] = class_field
        return Instance(labels)
