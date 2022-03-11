# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.apps import AppConfig
from transformers import AutoTokenizer, RobertaTokenizer
from transformers import TFRobertaModel
from transformers import BertTokenizer, TFBertModel
from transformers import TFXLNetModel
import tensorflow as tf


class ServerConfig(AppConfig):
    name = 'server'
    roberta_tokenizer = None
    bert_tokenizer = None
    XLnet_tokenizer = None
    roberta_model = None
    bert_model = None
    Xlnet_model = None

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = tf.reduce_mean(last_hidden_state, 1)

        return mean_last_hidden_state

    def create_model(self, model, max_len, pool=False):
        input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
        attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32')

        output = model([input_ids, attention_masks])

        if(pool == True):
            output = self.pool_hidden_state(output)

        else:
            output = output[1]

        output = tf.keras.layers.Dense(7, activation='softmax')(output)
        model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks], outputs=output)
        model.compile(tf.keras.optimizers.Adam(lr=1e-5),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def ready(self):
        max_len = 40
        # run on Roberta
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained(
            'roberta-large',
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True)  # Tokenizer

        self.roberta_model = TFRobertaModel.from_pretrained('roberta-large')
        self.roberta_model = self.create_model(self.roberta_model, max_len)
        self.roberta_model.load_weights("saved_models/Roberta_weights.h5")

        # run on Bert
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-large-cased',
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True)  # Tokenizer

        self.bert_model = TFBertModel.from_pretrained('bert-large-cased')
        self.bert_model = self.create_model(self.bert_model, max_len)
        self.bert_model.load_weights("saved_models/Bert.h5")

        # run on Xlnet
        self.XLnet_tokenizer = AutoTokenizer.from_pretrained(
            'xlnet-large-cased',
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True)

        self.Xlnet_model = TFXLNetModel.from_pretrained('xlnet-large-cased')
        self.Xlnet_model = self.create_model(self.Xlnet_model, max_len, pool=True)
        self.Xlnet_model.load_weights("saved_models/XLnet.h5")
