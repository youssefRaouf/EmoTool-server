# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.apps import AppConfig
from transformers import AlbertTokenizerFast, TFAlbertModel
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
        self.roberta_tokenizer = AlbertTokenizerFast.from_pretrained(
            'albert-base-v1',
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True)  # Tokenizer

        self.roberta_model = TFAlbertModel.from_pretrained('albert-base-v1')
        self.roberta_model = self.create_model(self.roberta_model, max_len)
        self.roberta_model.load_weights("saved_models/albert_v1_68_58.h5")
