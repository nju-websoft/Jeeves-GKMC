# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless requreader2d by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil

import torch
from torch import nn
import jieba
from json_reader import getFeature
from IR_Http import bm25_search_weight, bm25_search, bm25_search_weight_list
from bm25_score import get_bm25_score_word
import heapq

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly diffretrievernt (and gives slightly diffretrievernt results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_softmax = attention_probs

        # This is actually dropping out entreader2 tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_softmax


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states, attention_probs = layer_module(hidden_states, attention_mask)

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but thretriever is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly diffretrievernt from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrdretrieverdDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redreader2ct to the cache, if necessary
        resolved_archive_file = archive_file
        """
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        """
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or resolved_archive_file.startswith('//philly'):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        # config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config_file = '/'.join([serialization_dir, CONFIG_NAME])
        config = BertConfig.from_json_file(config_file)
        # logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidreader2ctional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension hretriever.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entreader2ly.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class WordWeightingLayer(nn.Module):
    def __init__(self, dim):
        super(WordWeightingLayer, self).__init__()
        self.W1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, 1)
        self.activate = nn.Tanh()
        self.activate2 = nn.Sigmoid()
        self.maxpooling = nn.AdaptiveMaxPool1d(1)

        self.attention_head_size = dim
        self.query = nn.Linear(dim, self.attention_head_size)
        self.key = nn.Linear(dim, self.attention_head_size)

    def forward(self, hidden_states, mask=None, word_set_idx=None):
        batch = hidden_states.size(0)
        seq_len = hidden_states.size(1)
        dim = hidden_states.size(2)
        h = hidden_states.view(-1, dim)
        h = self.W1(h)
        h = self.activate(h)

        h_aggregate, mask_aggregate = self.aggregate_word(word_set_idx, h, seq_len, dim)
        h_aggregate = self.w2(h_aggregate)
        h_aggregate = h_aggregate.view(batch, seq_len)

        mask_aggregate = 1 - mask_aggregate
        mask_aggregate = -10000.0 * mask_aggregate
        h_aggregate = h_aggregate.float() + mask_aggregate.float()
        return h_aggregate

    # 相同的词聚到一个表示
    def aggregate_word(self, word_set_idx, h, seq_len, dim):
        max_word_fre = 0
        for item_idx in word_set_idx:
            for idx in item_idx:
                if len(idx) > max_word_fre:
                    max_word_fre = len(idx)
        offset = 0
        step_size = seq_len
        select_idx = []
        mask = []
        padding_idx = [0 for i in range(max_word_fre)]

        for item_idx in word_set_idx:

            mask_item = []
            select_idx_seq = []
            for idx in item_idx:
                mask_item.append(1)
                select_idx_item = []
                for i in idx:
                    select_idx_item.append(offset * step_size + i + 1)
                while len(select_idx_item) < max_word_fre:
                    select_idx_item.append(0)
                select_idx_seq.append(select_idx_item)
            while len(select_idx_seq) < seq_len:
                select_idx_seq.append(padding_idx)
            select_idx.append(select_idx_seq)
            while len(mask_item) < seq_len:
                mask_item.append(0)
            mask.append(mask_item)
            offset += 1

        select_idx = torch.tensor(select_idx)
        mask = torch.tensor(mask)
        select_idx = select_idx.cuda()
        mask = mask.cuda()
        select_idx = select_idx.view(-1)
        h = h.view(-1, dim)
        h = torch.cat([h.new_zeros((1, dim)), h], dim=0)  # 增加一列用作填充的向量

        h_word_aggregate = h.index_select(0, select_idx)

        h_word_aggregate = h_word_aggregate.view(-1, max_word_fre, dim)
        h_word_aggregate = h_word_aggregate.transpose(dim0=1, dim1=2)
        h_word_aggregate_pooling = self.maxpooling(h_word_aggregate)
        h_word_aggregate_pooling = h_word_aggregate_pooling.squeeze()
        h_word_aggregate_pooling = h_word_aggregate_pooling.view(-1, seq_len, dim)
        return h_word_aggregate_pooling, mask


class FineSemanticMatch(nn.Module):
    def __init__(self, dim):
        super(FineSemanticMatch, self).__init__()
        dpda_layer = DPDALayear(dim)
        layer_num = 1
        self.p_b = nn.ModuleList([copy.deepcopy(dpda_layer) for _ in range(layer_num)])
        self.p_q = nn.ModuleList([copy.deepcopy(dpda_layer) for _ in range(layer_num)])
        self.p_o = nn.ModuleList([copy.deepcopy(dpda_layer) for _ in range(layer_num)])
        self.b_q = nn.ModuleList([copy.deepcopy(dpda_layer) for _ in range(layer_num)])
        self.b_o = nn.ModuleList([copy.deepcopy(dpda_layer) for _ in range(layer_num)])
        self.q_o = nn.ModuleList([copy.deepcopy(dpda_layer) for _ in range(layer_num)])

        self.linear = nn.Linear(2 * dim, dim)
        self.linear_all = nn.Linear(6 * dim, dim)
        self.relu = nn.ReLU()

        self.v1_w = nn.Linear(dim, dim)
        self.v2_w = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()

        self.lstm = nn.LSTM(dim, dim, 1, batch_first=True)

    def forward(self, p, b, q, o, mask_p, mask_b, mask_q, mask_o):

        for layer_module in self.p_b:
            p, b = layer_module(p, b, mask_p, mask_b)
        p_b = self.get_vector(p, b)

        for layer_module in self.p_q:
            p, q = layer_module(p, q, mask_p, mask_q)
        p_q = self.get_vector(p, q)

        for layer_module in self.p_o:
            p, o = layer_module(p, o, mask_p, mask_o)
        p_o = self.get_vector(p, o)

        for layer_module in self.b_q:
            b, q = layer_module(b, q, mask_b, mask_q)
        b_q = self.get_vector(b, q)

        for layer_module in self.b_o:
            b, o = layer_module(b, o, mask_b, mask_o)
        b_o = self.get_vector(b, o)

        for layer_module in self.q_o:
            q, o = layer_module(q, o, mask_q, mask_o)
        q_o = self.get_vector(q, o)

        vec = torch.cat([p_b, p_q, p_o, b_q, b_o, q_o], dim=1)  # 6*dim
        output = vec
        output = self.linear_all(vec)
        output = self.relu(output)
        return output

    def get_vector(self, v1, v2):
        v1, _ = torch.max(v1, dim=1)
        v2, _ = torch.max(v2, dim=1)
        v1, v2 = v1.squeeze(), v2.squeeze()
        p_b = torch.cat([v1, v2], dim=1)  # 2*dim
        p_b = self.linear(p_b)  # dim
        p_b = self.relu(p_b)
        return p_b


class DPDALayear(nn.Module):
    def __init__(self, dim):
        super(DPDALayear, self).__init__()
        self.W_p = nn.Linear(2 * dim, dim)
        self.W_q = nn.Linear(2 * dim, dim)

    def forward(self, P, Q, p_mask=None, q_mask=None):
        P_ori = P
        Q_ori = Q
        A = torch.matmul(P, Q.transpose(dim0=1, dim1=2))  # l_p, l_q


        if p_mask is not None:
            p_mask = p_mask.float()
            p_mask = 1 - p_mask
            p_mask = p_mask * -10000.0
            p_mask = p_mask.unsqueeze(dim=2)
            p_mask = p_mask.expand_as(A)
            A = A + p_mask

        if q_mask is not None:
            q_mask = q_mask.float()
            q_mask = 1 - q_mask
            q_mask = q_mask * -10000.0
            q_mask = q_mask.unsqueeze(dim=1)
            q_mask = q_mask.expand_as(A)
            A = A + q_mask

        A_q = torch.softmax(A, dim=2)  # l_p, l_q
        A_p = torch.softmax(A.transpose(dim0=1, dim1=2), dim=2)  # l_q, l_p

        P_q = torch.matmul(A_q, Q)  # l_p, dim
        Q_p = torch.matmul(A_p, P)  # l_q, dim

        P_t = torch.cat([P_q, P], dim=2)  # l_p, 2*dim
        Q_t = torch.cat([Q_p, Q], dim=2)  # l_q, 2*dim

        Q = torch.matmul(A_p, P_t)  # l_q, 2*dim
        P = torch.matmul(A_q, Q_t)  # l_p, 2*dim

        P = P_ori + self.W_p(P)  # l_p, dim
        Q = Q_ori + self.W_q(Q)  # l_q, dim

        return P, Q


class SemanticMatchLayer(nn.Module):
    def __init__(self, dim):
        super(SemanticMatchLayer, self).__init__()
        self.trans_linear = nn.Linear(dim, dim)
        self.map_linear = nn.Linear(2 * dim, dim)
        self.gru = nn.GRU(2 * dim, dim, 2)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, p, q, p_mask, q_mask, subword_weight_p_p, subword_weight_p_q):
        subword_weight_p_p = subword_weight_p_p.unsqueeze(dim=2)
        subword_weight_p_q = subword_weight_p_q.unsqueeze(dim=1)
        weight_metrix = torch.matmul(subword_weight_p_p, subword_weight_p_q)

        A = p.bmm(torch.transpose(q, 1, 2))
        # A = A * weight_metrix
        if p_mask is not None:
            p_mask = p_mask.float()
            p_mask = 1 - p_mask
            p_mask = p_mask * -10000.0
            p_mask = p_mask.unsqueeze(dim=2)
            p_mask = p_mask.expand_as(A)
            A = A + p_mask

        if q_mask is not None:
            q_mask = q_mask.float()
            q_mask = 1 - q_mask
            q_mask = q_mask * -10000.0
            q_mask = q_mask.unsqueeze(dim=1)
            q_mask = q_mask.expand_as(A)
            A = A + q_mask

        att_norm_p_q = torch.softmax(A, dim=2)  # batch, p, q
        S_p = torch.matmul(att_norm_p_q, q)  # batch, p, dim

        att_norm_q_p = torch.softmax(A.transpose(dim0=1, dim1=2), dim=2)
        S_q = torch.matmul(att_norm_q_p, p)  # batch, q, dim
        C_p = torch.matmul(att_norm_p_q, S_q)  # batch, p, dim
        C_p = torch.cat([C_p, S_p], dim=2)  # batch, p, 2*dim

        output, _ = self.gru(C_p)  # batch, p, dim

        output_att = self.tanh(self.linear2(self.linear1(output)))
        output_att = output_att.squeeze()  # batch, p
        output_att = torch.softmax(output_att, dim=1)
        output_att = output_att.unsqueeze(dim=2)
        output_att = output_att.expand_as(output)

        output = output * output_att
        output_pool = torch.sum(output, dim=1)

        return output_pool


class SelfAttentionFuseLayer(nn.Module):
    def __init__(self, dim):
        super(SelfAttentionFuseLayer, self).__init__()
        self.W_7 = nn.Linear(dim, dim)
        self.w_8 = nn.Linear(dim, 1)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        h1 = self.W_7(hidden_states)
        h1 = self.activation(h1)
        h2 = self.w_8(h1)
        h2 = self.activation(h2)
        h2 = h2.squeeze()
        a = torch.softmax(h2, dim=1)
        a = a.unsqueeze(dim=2)
        a = a.expand_as(hidden_states)
        hidden_states = hidden_states * a
        hidden_states = torch.sum(hidden_states, dim=1)
        hidden_states = hidden_states.squeeze()
        return hidden_states


class Jeeves_Model(PreTrainedBertModel):
    def __init__(self, config, num_choices=4,
                 max_seq_length=128,
                 tau=200, k_train=10, k_test=5, k2_num=2, index_map=None, tokenizer=None):
        super(Jeeves_Model, self).__init__(config)

        self.tokenizer = tokenizer

        self.tau = tau
        self.k_num_train = k_train
        self.k_num_test = k_test
        self.k2_num = k2_num

        self.index_map = index_map

        self.max_seq_length = max_seq_length
        self.num_choices = num_choices
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)
        self.criterion = nn.CrossEntropyLoss()
        self.word_weighting = WordWeightingLayer(config.hidden_size)

        self.semantic_match = FineSemanticMatch(config.hidden_size)
        self.self_attention_fuse = SelfAttentionFuseLayer(config.hidden_size)

        self.choice_linear = nn.Linear(config.hidden_size, 1) #w_5

        self.W_3 = nn.Linear(self.tau, self.tau)
        self.w_4 = nn.Linear(self.tau, 1)

        self.w_9 = nn.Linear(config.hidden_size * 1,
                                          config.hidden_size * 1)
        self.w_10 = nn.Linear(config.hidden_size * 1, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, input_ids, input_mask, segment_ids, q_ids=None, answers=None, have_negative_word=None,
                orig_to_token_split_idx=None, orig_to_token_split=None, word_set_idx=None, word_set=None,
                paragraphs=None, scenarios=None, questions=None, options=None,
                scenario_range_subword=None, question_range_subword=None, option_range_subword=None,
                bm25_feature=None, p_labels=None, have_p_labels=None, p_ids=None,
                do_train=True, do_eval=False, do_pre_finetune=False, retriever=True, reader1=True, reader2=True, p=False):

        if do_pre_finetune is True:
            hidden_states_q, pooled_output_q = self.bert(input_ids, segment_ids, input_mask,
                                                         output_all_encoded_layers=False)
            pooled_output_q = self.dropout(pooled_output_q)
            u_l = self.choice_linear(pooled_output_q)
            u_l = u_l.view(-1, 4)
            if do_train:
                loss = self.criterion(u_l, answers)
                return u_l, loss
            else:
                return u_l
        if do_train:
            reader1_num = self.k_num_train
        else:
            reader1_num = self.k_num_test

        seq_len = input_ids.size(1)
        tmp = input_ids.size(0)
        batch_size = tmp // self.num_choices
        hidden_states_q, pooled_output_q = self.bert(input_ids, segment_ids, input_mask, output_all_encoded_layers=False)
        dim = hidden_states_q.size(-1)
        hidden_states_q = self.dropout(hidden_states_q)

        word_weights = self.word_weighting(hidden_states_q, input_mask, word_set_idx)
        word_weights_soft = torch.softmax(word_weights, dim=-1)

        if do_eval:
            ir_size = 300
            p_ids, paragraphs, bm25_feature = self.get_ir_feature(word_weights_soft, word_set, ir_size, seq_len,
                                                                  self.index_map, scenarios, questions, options)
            bm25_feature = bm25_feature.view(batch_size * self.num_choices, ir_size, seq_len)
            word_weight_aggree_exp = word_weights_soft.unsqueeze(dim=1)
            word_weight_aggree_exp = word_weight_aggree_exp.expand_as(bm25_feature)
            p_ids, paragraphs, bm25_feature = self.first_fillter(paragraphs, p_ids, bm25_feature,
                                                                 word_weight_aggree_exp,
                                                                 batch_size, ir_size, self.tau,
                                                                 self.max_seq_length)

        bm25_feature = bm25_feature.view(batch_size * self.num_choices, self.tau, seq_len)
        word_weights_soft_exp = word_weights_soft.unsqueeze(dim=1)
        word_weights_soft_exp = word_weights_soft_exp.expand_as(bm25_feature)
        z_l_item = bm25_feature.float() * word_weights_soft_exp
        z_l = torch.sum(z_l_item, dim=2)
        z_l_sorted, z_l_idx = torch.sort(z_l, dim=1, descending=True)  # batch_size * self.num_choices, retriever_num
        z_l_sorted = z_l_sorted.unsqueeze(dim=2)
        z_l_sorted = z_l_sorted.squeeze()
        score_r = self.W_3(z_l_sorted)
        score_r = self.tanh(score_r)
        score_r = self.w_4(score_r)
        score_r = score_r.view(-1, self.num_choices)
        score_r_soft = torch.softmax(score_r, dim=1)

        p_input_ids, p_input_mask, p_segment_ids, \
        p_paragraph_range_subword, p_scenario_range_subword, \
        p_question_range_subword, p_option_range_subword, p_labels_reader1 \
            = self.get_p_reader1(z_l, p_labels, paragraphs,
                             scenarios, questions, options, batch_size, reader1_num, word_set, word_weights_soft)

        p_labels = p_labels.view(-1, reader1_num)
        hidden_states_g, pooled_output_g = self.bert(p_input_ids, p_segment_ids, p_input_mask,
                                                     output_all_encoded_layers=False)
        hidden_states_g = self.dropout(hidden_states_g)
        pooled_output_g = self.dropout(pooled_output_g)
        u_l = self.choice_linear(pooled_output_g)
        u_l = u_l.view(-1, reader1_num)
        score_f = torch.sum(u_l, dim=1)
        score_f = score_f.view(-1, self.num_choices)
        score_f_soft = torch.softmax(score_f, dim=1)



        hidden_states_g_p, mask_g_p = self.split_sequence(hidden_states_g, p_paragraph_range_subword, useSep=True)
        hidden_states_g_s, mask_g_s = self.split_sequence(hidden_states_g, p_scenario_range_subword, useSep=True)
        hidden_states_g_q, mask_g_q = self.split_sequence(hidden_states_g, p_question_range_subword, useSep=True)
        hidden_states_g_o, mask_g_o = self.split_sequence(hidden_states_g, p_option_range_subword, useSep=True)

        select_index = self.get_reader2_p_index(u_l, self.k2_num, reader1_num)
        select_index = select_index.view(-1)
        hidden_states_g_p, mask_g_p = hidden_states_g_p[select_index], mask_g_p[select_index]
        hidden_states_g_s, mask_g_s = hidden_states_g_s[select_index], mask_g_s[select_index]
        hidden_states_g_q, mask_g_q = hidden_states_g_q[select_index], mask_g_q[select_index]
        hidden_states_g_o, mask_g_o = hidden_states_g_o[select_index], mask_g_o[select_index]


        # have_negative_word_exp = have_negative_word.unsqueeze(dim=1)
        # have_negative_word_exp = have_negative_word_exp.expand_as(choice_score)
        # choice_score = choice_score.float() * have_negative_word_exp.float()
        # choice_score = choice_score * have_negative_word_exp

        f_l = self.semantic_match(hidden_states_g_p, hidden_states_g_s, hidden_states_g_q,hidden_states_g_o,
                                           mask_g_p, mask_g_s, mask_g_q, mask_g_o)

        f_l = f_l.view(-1, self.k2_num, dim)
        f = self.self_attention_fuse(f_l)
        f = f.view(-1, self.num_choices, dim)
        f = self.w_9(f)
        f = self.relu(f)
        score_d = self.w_10(f)
        score_d = score_d.view(-1, self.num_choices)
        score_d_soft = torch.softmax(score_d, dim=1)

        p_labels = p_labels.view(batch_size * self.num_choices, self.tau)
        if do_train:

            loss_retriever = self.criterion(score_r, answers)
            loss_reader1 = self.criterion(score_f, answers)
            loss_reader2 = self.criterion(score_d, answers)

            have_p_labels = have_p_labels.view(-1)
            z_l = z_l.view(-1, self.tau)
            active_choice = have_p_labels == 1
            p_labels_active = p_labels[active_choice]
            z_l_active = z_l[active_choice]
            pos_num, _ = z_l_active.size()
            if pos_num == 0:
                loss_retriever_p = None
            else:
                loss_retriever_p = self.listnet_loss(z_l_active, p_labels_active)

            u_l_active = u_l.view(batch_size, self.num_choices * reader1_num)
            p_labels_reader1_active = p_labels_reader1.view(batch_size, self.num_choices * reader1_num)

            loss_reader1_p = self.cross_entropy_loss(u_l_active, p_labels_reader1_active)

            loss = torch.tensor(0, dtype=torch.float).cuda()
            # loss = 0 * loss_reader1
            if retriever:
                loss += loss_retriever
            if reader1:
                loss += loss_reader1
            if reader2:
                loss += loss_reader2
            if p:
                loss += loss_retriever_p + loss_reader1_p
            return loss
        else:
            p_labels = p_labels.view(-1, self.tau)
            return score_r_soft, score_f_soft, score_d_soft, p_labels, p_ids, z_l, paragraphs

    def get_reader2_p_index(self, scores, reader2_num, reader1_num):
        scores = scores.view(-1, reader1_num)
        scores = scores.cpu().detach().numpy().tolist()
        select_index = []
        count = 0
        count1 = 0
        for score in scores:
            data = heapq.nlargest(reader2_num, enumerate(score), key=lambda x: x[1])
            max_index, vals = zip(*data)
            max_index = list(max_index)
            select = []
            count1 += len(max_index)
            for i in range(reader1_num):
                if i in max_index:
                    select.append(True)
                    count += 1
                else:
                    select.append(False)
            select_index.append(select)
        select_index = torch.tensor(select_index)
        select_index = select_index.cuda()
        return select_index

    def get_ir_feature(self, weight_set, word_set, max_result_num, max_seq_length, index_map, backgrounds, questions,
                       options):
        weight_set = weight_set.cpu().detach().numpy().tolist()
        bm25_features = []
        paragraphs = []
        p_ids = []
        for words, weights, background, question, option in zip(word_set, weight_set, backgrounds, questions, options):
            weight_all = []
            weight_map = {}
            for i, word in enumerate(words):
                weight = weights[i]
                weight_all.append(weight)
                # weight_all.append(1)
                weight_map[word] = weight
            query_tokens = cut(background) + cut(question) + cut(option)
            query_tokens_weight = []
            query_tokens_fillter = []
            for query_token in query_tokens:
                if query_token in weight_map:
                    query_tokens_fillter.append(query_token)
                    query_tokens_weight.append(weight_map[query_token])

            p_ids_retrieval, contents, bm25_scores = bm25_search_weight_list(words, weight_all, max_result_num)

            paragraphs += contents
            p_ids.append(p_ids_retrieval)

            for content in contents:
                tokens_paragraph = cut(content)
                bm25_list = []
                for q in words:
                    score = get_bm25_score_word(q, None, tokens_paragraph, index_map)
                    bm25_list.append(score)

                while (len(bm25_list) < max_seq_length):
                    bm25_list.append(0)
                bm25_features.append(bm25_list)

        bm25_features = torch.tensor(bm25_features, dtype=torch.float)
        bm25_features = bm25_features.cuda()
        p_ids = torch.tensor(p_ids, dtype=torch.long)
        p_ids = p_ids.cuda()

        return p_ids, paragraphs, bm25_features

    def first_fillter(self, paragraphs, p_ids, bm25_features, word_weight, batch_size, ir_size, top_k, max_len):
        bm25_score_item = bm25_features.float() * word_weight
        bm25_score = torch.sum(bm25_score_item, dim=2)
        bm25_score = bm25_score.view(batch_size * self.num_choices, ir_size)
        bm25_score = bm25_score.cpu().detach().numpy().tolist()
        step_size = ir_size
        i = 0
        idx_select = []
        paragraph_list = []
        for score in bm25_score:
            data = heapq.nlargest(top_k, enumerate(score), key=lambda x: x[1])
            max_index, vals = zip(*data)
            max_num_index_list = list(max_index)
            for idx in max_num_index_list:
                idx_select.append(i * step_size + idx)
                paragraph = paragraphs[i * step_size + idx]
                paragraph_list.append(paragraph)
            i += 1
        idx_select = torch.tensor(idx_select)
        idx_select = idx_select.cuda()
        p_ids = p_ids.view(-1, 1)
        p_ids = p_ids.index_select(0, idx_select)
        p_ids = p_ids.view(-1, top_k)

        bm25_features = bm25_features.view(-1, max_len)
        bm25_features = bm25_features.index_select(0, idx_select)
        bm25_features = bm25_features.view(-1, max_len)

        return p_ids, paragraph_list, bm25_features

    def get_p_reader1(self, z_ls, p_labels, paragraphs, scenarios, questions, options, batch_size,
                        top_k, word_sets, word_weights):

        z_ls = z_ls.view(batch_size * self.num_choices, self.tau)

        step_size = self.tau
        i = 0
        idx_selext = []
        input_ids, input_mask, segment_ids = [], [], []
        paragraph_range_subword, scenario_range_subword, question_range_subword, option_range_subword = [], [], [], []

        for z_l, background, question, option, word_set, word_weight in zip(z_ls, scenarios, questions,
                                                                            options, word_sets, word_weights):

            z_l = z_l.cpu().detach().numpy().tolist()
            data = heapq.nlargest(top_k, enumerate(z_l), key=lambda x: x[1])
            max_index, vals = zip(*data)
            max_num_index_list = list(max_index)
            paragraph_list = []
            score_list = []
            for idx in max_num_index_list:
                score_list.append(z_l[idx])
                idx_selext.append(i * step_size + idx)
                paragraph = paragraphs[i * step_size + idx]
                paragraph_list.append(paragraph)
                option_feature = getFeature(paragraph, background, question, option, self.tokenizer,
                                            self.max_seq_length)
                input_ids.append(option_feature.input_ids)
                input_mask.append(option_feature.input_mask)
                segment_ids.append(option_feature.segment_ids)
                paragraph_range_subword.append(option_feature.paragraph_range_subword)
                scenario_range_subword.append(option_feature.scenario_range_subword)
                question_range_subword.append(option_feature.question_range_subword)
                option_range_subword.append(option_feature.option_range_subword)
            i += 1

        idx_selext = torch.tensor(idx_selext)
        idx_selext = idx_selext.cuda()
        p_labels = p_labels.view(-1, 1)
        p_labels = p_labels.index_select(0, idx_selext)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        input_ids = input_ids.cuda()
        input_mask = input_mask.cuda()
        segment_ids = segment_ids.cuda()

        return input_ids, input_mask, segment_ids, \
               paragraph_range_subword, scenario_range_subword, \
               question_range_subword, option_range_subword, p_labels


    def cross_entropy_loss(self, predicts, label):
        label = label.float()
        predicts_logits = torch.softmax(predicts, dim=1)
        loss1 = - torch.sum(label * torch.log(predicts_logits), dim=1)
        loss2 = - torch.sum((1 - label) * torch.log(1 - predicts_logits), dim=1)
        loss = torch.cat([loss1, loss2], dim=0)
        loss = torch.mean(loss)
        loss = loss * 2
        return loss

    def listnet_loss(self, predicts, label):
        label = label.float()
        label = label * 2
        predicts = predicts.contiguous()
        label = label.contiguous()
        P_y_i = torch.softmax(predicts, dim=1)
        P_label = torch.softmax(label, dim=1)
        loss = - torch.sum(P_label * torch.log(P_y_i), dim=1)
        loss = torch.mean(loss)
        return loss

    def rank_loss(self, predicts, label):
        label = label.float()

        P_y_i = torch.softmax(predicts, dim=1)
        P_y_i = - torch.log(P_y_i)
        P_y_i = P_y_i * label
        P_y_i = torch.sum(P_y_i, dim=1)
        num = torch.sum(label, dim=1)
        loss = P_y_i / num
        loss = torch.mean(loss)
        return loss


    def split_sequence(self, bert_output_word_q, background_range, useSep=True, set_max_len=None):
        useSep_offset = 0
        if useSep:
            useSep_offset = 1
        batch = bert_output_word_q.size(0)
        seq_len = bert_output_word_q.size(1)
        dim = bert_output_word_q.size(2)

        max_len = 0
        for b_range in background_range:
            q_beg = b_range[0]
            c_end = b_range[1]
            question_len = c_end - q_beg + 1
            if question_len > max_len:
                max_len = question_len
        if useSep:
            max_len += 1
        if set_max_len is not None:
            if set_max_len > max_len:
                max_len = set_max_len
        bert_output_word_q = bert_output_word_q.reshape(-1, dim)
        index_select = []
        offset = 0
        step_size = seq_len
        mask = []
        tmp = 0
        for index in range(batch):
            b_range = background_range[index]
            q_beg = b_range[0]
            c_end = b_range[1] + useSep_offset
            index_select_item = []
            mask_item = []
            for i in range(q_beg, c_end + 1):
                index_select_item.append(offset * step_size + i + 1)
                if offset * step_size + i + 1 > tmp:
                    tmp = offset * step_size + i + 1
                mask_item.append(1)
            while len(index_select_item) < max_len:
                index_select_item.append(0)
                mask_item.append(0)
            index_select += index_select_item
            mask.append(mask_item)
            offset += 1

        index_select = torch.tensor(index_select, dtype=torch.long)
        index_select = index_select.cuda()
        mask = torch.tensor(mask, dtype=torch.long)
        mask = mask.cuda()
        sequence_output = bert_output_word_q.reshape(-1, dim)
        sequence_output = torch.cat([sequence_output.new_zeros((1, dim), dtype=torch.float), sequence_output],
                                    dim=0)
        sequence_new = sequence_output.index_select(0, index_select)
        sequence_new = sequence_new.view(batch, max_len, dim)

        return sequence_new, mask


def cut(content):
    content_cut = jieba.cut(content)
    return list(content_cut)
