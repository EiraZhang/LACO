from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
import collections
import collections
import csv
import os
import time
import sys
import tensorflow as tf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ie.src.bert import modeling
from ie.src.bert import optimization
from ie.src.bert import tokenization
from ie.src.bert import tf_metrics
from ie.evaluate_main import evaluate_main
from common.global_config import get_ie_config, get_job_id, get_schema_set, get_train_para, get_pretrain_model_path
import random

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "model_idx_info", None,
    "model_idx_info. ")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_path_dir", None,
    "init path.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "train_and_evaluate", False,
    "Whether to use train_and_evaluate")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test_with_results", False, "Whether test with results.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_float(
    "layerwise_lr_decay", 0.8,
    "# if > 0, the learning rate for a layer i lr * lr_decay^(depth - max_depth) i.e., "
    "shallower layers have lower learning rates"
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_list(
    "labels_list", [],
    "labels which need to train.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 token_label_ids,
                 label_ids,
                 fit_labelspace_positions,
                 fit_docspace_positions,
                 pair,
                 pair_target,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_label_ids = token_label_ids
        self.label_ids = label_ids
        self.fit_labelspace_positions = fit_labelspace_positions
        self.fit_docspace_positions = fit_docspace_positions
        self.pair = pair
        self.pair_target = pair_target
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Multi_Label_Classification_Processor(DataProcessor):
    def __init__(self):
        self.language = "zh"

    def get_examples(self, data_dir):
        with open(os.path.join(data_dir, "token_in.txt"), encoding='utf-8') as token_in_f:
            with open(os.path.join(data_dir, "predicate_out.txt"), encoding='utf-8') as predicate_out_f:
                token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
                predicate_label_list = [seq.replace("\n", '') for seq in predicate_out_f.readlines()]
                assert len(token_in_list) == len(predicate_label_list)
                examples = list(zip(token_in_list, predicate_label_list))
                return examples

    def get_train_examples(self, data_dir):
        examples = self.get_examples(os.path.join(data_dir, "train"))
        return self._create_example(examples, "train")

    def get_dev_examples(self, data_dir):
        examples= self.get_examples(os.path.join(data_dir, "valid"))
        return self._create_example(examples, "valid")

    def get_test_examples(self, data_dir):
        with open(os.path.join(data_dir, os.path.join("test", "token_in.txt")), encoding='utf-8') as token_in_f:
            token_in_list = [seq.replace("\n", '') for seq in token_in_f.readlines()]
            examples = token_in_list
            return self._create_example(examples, "test")

    def get_token_labels(self):
        token_labels = ["UNLABEL", "LABEL"]  # 0,1
        return token_labels

    def get_labels(self):
            return FLAGS.labels_list
    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_str = line
                if len(FLAGS.labels_list) == 0:
                    predicate_label_str = ''
                else:
                    label = FLAGS.labels_list
                    predicate_label_str = label[0]
            else:
                text_str = line[0]
                predicate_label_str = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_str, text_b=None,
                             label=predicate_label_str))

        return examples


def convert_single_example(ex_index, example, token_label_list, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            token_label_ids=[0] * max_seq_length,
            label_ids=[0] * len(label_list),
            fit_labelspace_positions=[0] * len(label_list),
            fit_docspace_positions=[0] * (max_seq_length-len(label_list)),
            pair = [0]*2,
            pair_target = [0]*2,
            is_real_example=False)

    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[label] = i

    label_map = {}

    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = example.text_a.split(" ")

    bias = 1
    label_trans_token = {}
    for (i, label) in enumerate(label_list):
        if i+bias<100:
            label_trans_token[label] = i + bias
        else:
            label_trans_token[label] = i + bias + 4

    label_list = example.label.split(" ")
    label_ids = _predicate_label_to_id(label_list, label_map)
    right_labels = []
    wrong_labels = []
    for label_id in range(0,len(label_ids)):
        if label_ids[label_id]==1:
            right_labels.append(label_id)
        else:
            wrong_labels.append(label_id)

    right_pair = list(itertools.combinations(right_labels, 2))

    contrast_dict = {}

    for pair in right_pair:
        contrast_dict[pair]=[0,1]
    for i in range(0,int(len(right_pair)*2)):
        r = random.sample(right_labels,1)[0]
        w =random.sample(wrong_labels,1)[0]
        contrast_dict[(r,w)]=[1,0]
    token_b, token_b_ids, token_b_label = _general_token_b_and_seq_label(label_list, label_trans_token)
    if token_b_ids:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, token_b_ids, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    token_label_ids = []
    fit_labelspace_positions = []
    fit_docspace_positions = []
    doc_idx = 0

    tokens.append("[CLS]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["UNLABEL"])
    fit_docspace_positions.append(doc_idx)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
        token_label_ids.append(token_label_map["UNLABEL"])
        doc_idx += 1
        fit_docspace_positions.append(doc_idx)

    tokens.append("[SEP]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["UNLABEL"])
    doc_idx += 1
    fit_docspace_positions.append(doc_idx)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    lsp_idx = len(input_ids)

    if token_b_ids:
        for tk, tbi, tbl in zip(token_b, token_b_ids, token_b_label):
            tokens.append(tk)
            input_ids.append(tbi)
            segment_ids.append(1)
            token_label_ids.append(token_label_map[tbl])
            fit_labelspace_positions.append(lsp_idx)
            lsp_idx += 1

        tokens.append("[SEP]")
        input_ids.append(tokenizer.convert_tokens_to_ids(["[SEP]"])[0])
        segment_ids.append(0)
        token_label_ids.append(token_label_map["UNLABEL"])
        doc_idx = lsp_idx
        fit_docspace_positions.append(doc_idx)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        tokens.append("[Padding]")
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        token_label_ids.append(token_label_map["UNLABEL"])
        doc_idx += 1
        fit_docspace_positions.append(doc_idx)


    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(token_label_ids) == max_seq_length
    assert (len(fit_docspace_positions)+len(fit_labelspace_positions)) == max_seq_length


    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("token_label_ids: %s" % " ".join([str(x) for x in token_label_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("fit_labelspace_positions: %s" % " ".join([str(x) for x in fit_labelspace_positions]))
        tf.logging.info("fit_docspace_positions: %s" % " ".join([str(x) for x in fit_docspace_positions]))
        tf.logging.info(contrast_dict)
        tf.logging.info("len of (fit_labelspace_positions): %s" % len(fit_labelspace_positions))

    # feature = InputFeatures(
    #     input_ids=input_ids,
    #     input_mask=input_mask,
    #     segment_ids=segment_ids,
    #     token_label_ids=token_label_ids,
    #     label_ids=label_ids,
    #     fit_labelspace_positions=fit_labelspace_positions,
    #     fit_docspace_positions=fit_docspace_positions,
    #     is_real_example=True)
    # return feature
    feature_list=[]
    for pair in contrast_dict.keys():
        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            token_label_ids=token_label_ids,
            label_ids=label_ids,
            fit_labelspace_positions=fit_labelspace_positions,
            fit_docspace_positions=fit_docspace_positions,
            pair = list(pair),
            pair_target=list(contrast_dict[pair]),
        #    pair = list([0,1]),
        #    pair_target= list([0,1]),
            is_real_example=True)
        feature_list.append(feature)
    a = random.sample(feature_list, 1) 
    return a

def _predicate_label_to_id(predicate_label, predicate_label_map):
    predicate_label_map_length = len(predicate_label_map)
    predicate_label_ids = [0] * predicate_label_map_length

    for label in predicate_label:
        predicate_label_ids[predicate_label_map[label]] = 1
    return predicate_label_ids

def _general_token_b_and_seq_label(predicate_label, label_trans_token):
    token_b = []
    token_b_ids = []
    seq_label_token = []
    for k, v in label_trans_token.items():
        token_b.append(k)
        token_b_ids.append(v)
        if k not in predicate_label:
            seq_label_token.append("UNLABEL")
        else:
            seq_label_token.append("LABEL")
    return token_b, token_b_ids, seq_label_token

def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_offsets = tf.cast(flat_offsets, tf.int32)
  positions = tf.cast(positions, tf.int32)
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

'''
def _general_token_b_and_seq_label_random(predicate_label, label_trans_token):
    ture_label_num = len(predicate_label)
    label_length = len(label_trans_token)
    max_token_b_length = max(list([random.randint(int(label_length / 2), label_length), ture_label_num]))
    token_b = []
    token_b_ids = []
    seq_label_token = []
    for label in predicate_label:
        token_b_ids.append(label_trans_token[label])
        seq_label_token.append("LABEL")
        token_b.append(label)
    unrelated_label = []
    for k, v in label_trans_token.items():
        if k not in predicate_label:
            unrelated_label.append(k)
    random.shuffle(unrelated_label)
    i = 0
    while len(token_b_ids) < max_token_b_length:
        token_b.append(unrelated_label[i])
        token_b_ids.append(label_trans_token[unrelated_label[i]])
        seq_label_token.append("UNLABEL")
        i += 1
    return token_b, token_b_ids, seq_label_token
'''

def file_based_convert_examples_to_features(
        examples, token_label_list, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature_list = convert_single_example(ex_index, example, token_label_list, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        for i in feature_list:
            features = collections.OrderedDict()
            #print("input_ids",i.input_ids)
            features["input_ids"] = create_int_feature(i.input_ids)
            features["input_mask"] = create_int_feature(i.input_mask)
            features["segment_ids"] = create_int_feature(i.segment_ids)
            features["token_label_ids"] = create_int_feature(i.token_label_ids)
            features["label_ids"] = create_int_feature(i.label_ids)
            features["fit_labelspace_positions"] = create_int_feature(i.fit_labelspace_positions)
            features["fit_docspace_positions"] = create_int_feature(i.fit_docspace_positions)
            features["pair"]= create_int_feature(i.pair)
            features["pair_target"] = create_int_feature(i.pair_target)
            features["is_real_example"] = create_int_feature(
                [int(i.is_real_example)])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        # features = collections.OrderedDict()
        # features["input_ids"] = create_int_feature(feature.input_ids)
        # features["input_mask"] = create_int_feature(feature.input_mask)
        # features["segment_ids"] = create_int_feature(feature.segment_ids)
        # features["token_label_ids"] = create_int_feature(feature.token_label_ids)
        # features["label_ids"] = create_int_feature(feature.label_ids)
        # features["fit_labelspace_positions"] = create_int_feature(feature.fit_labelspace_positions)
        # features["fit_docspace_positions"] = create_int_feature(feature.fit_docspace_positions)
        # features["is_real_example"] = create_int_feature(
        #     [int(feature.is_real_example)])

        # tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        # writer.write(tf_example.SerializeToString())
    writer.close()
def hidden2tag(hiddenlayer,numclass):
    linear = tf.keras.layers.Dense(units=numclass,use_bias=True, activation=tf.nn.relu)
    return linear(hiddenlayer)

def file_based_input_fn_builder(input_file, seq_length, label_length,
                                is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "token_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([label_length], tf.int64),
        "fit_labelspace_positions": tf.FixedLenFeature([label_length], tf.int64),
        "fit_docspace_positions": tf.FixedLenFeature([seq_length-label_length], tf.int64),
        "pair": tf.FixedLenFeature([2], tf.int64),
        "pair_target":tf.FixedLenFeature([2], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        ##cut sentence frist deal with tokens_a
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 token_label_ids, labels, positions, doc_positions, num_token_labels, num_labels, use_one_hot_embeddings,pair, pair_target):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    token_label_output_layer = model.get_sequence_output()
    token_label_hidden_size = token_label_output_layer.shape[-1].value
    doc_seq_length = token_label_output_layer.shape[-2].value - num_labels
    doc_output_layer = gather_indexes(token_label_output_layer, doc_positions)
    doc_output_layer = tf.reshape(doc_output_layer, [-1, doc_seq_length, token_label_hidden_size])

    token_label_output_layer = gather_indexes(token_label_output_layer, positions)
    token_label_output_layer = tf.reshape(token_label_output_layer, [-1, num_labels, token_label_hidden_size])

    token_label_output_weight = tf.get_variable(
        "token_label_output_weight", [num_labels, token_label_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    token_label_output_bias = tf.get_variable(
        "token_label_output_bias", [num_labels], initializer=tf.zeros_initializer())
    contrast_output_weight_1 = tf.get_variable(
        "contrast_output_weight_1", [64, token_label_hidden_size*2],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    contrast_output_bias_1 = tf.get_variable(
        "contrast_output_bias_1", [64], initializer=tf.zeros_initializer())
    contrast_output_weight_2 = tf.get_variable(
        "contrast_output_weight_2", [2, 64],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    contrast_output_bias_2 = tf.get_variable(
        "contrast_output_bias_2", [2], initializer=tf.zeros_initializer())

    def joint_embedding(doc_output_layer, token_label_output_layer,pair):
        x_emb_norm = tf.nn.l2_normalize(doc_output_layer, axis=2)
        w_class_norm = tf.nn.l2_normalize(token_label_output_layer, axis=2)
        w_class_norm = tf.transpose(w_class_norm, [0, 2, 1])
        G = tf.matmul(x_emb_norm, w_class_norm)
        G = tf.expand_dims(G, axis=-1)
        label_length = token_label_output_layer.shape[-2].value
        Att_v = tf.contrib.layers.conv2d(G, num_outputs=label_length, kernel_size=[10, 10], padding='SAME',
                                         activation_fn=tf.nn.relu)
        Att_v = tf.reduce_max(Att_v, axis=-1, keepdims=True)
        Att_v = tf.squeeze(Att_v)
        Att_v = tf.reduce_max(Att_v, axis=-1, keepdims=True)
        Att_v_tan = tf.tanh(Att_v)
        x_emb_norm = tf.squeeze(x_emb_norm)
        x_att = tf.multiply(x_emb_norm, Att_v_tan)
        H_enc = tf.reduce_sum(x_att, axis=1)
        two_w_class_norm = gather_indexes(token_label_output_layer, pair)#32,2,768
        two_w_class_norm = tf.reshape(two_w_class_norm,[-1,2,768])
        two_w_class_norm = tf.transpose(two_w_class_norm , [0, 2, 1])#32,768,2
        G_pair_a = two_w_class_norm[:,:,0]
        G_pair_a = tf.reshape(G_pair_a,[-1,768])
        G_pair_b = two_w_class_norm[:, :, 1]
        G_pair_b = tf.reshape(G_pair_b, [-1, 768])
        a_b_enc = tf.concat([G_pair_a , G_pair_b],-1)
        return H_enc, a_b_enc

    H_enc, a_b_enc = joint_embedding(doc_output_layer, token_label_output_layer,pair)
    with tf.variable_scope("token_label_loss"):
        if is_training:
            H_enc = tf.nn.dropout(H_enc, keep_prob=0.9)
            a_b_enc = tf.nn.dropout(a_b_enc, keep_prob=0.9)
        logits_wx = tf.matmul(H_enc, token_label_output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits_wx, token_label_output_bias)
        probabilities = tf.sigmoid(logits)
        label_ids = tf.cast(labels, tf.float32)
        per_example_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_ids), axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        predict_ids = tf.cast(probabilities > 0.5, tf.int32)
        logits_pair = hidden2tag(a_b_enc , 2)
        contrast_probabilities = tf.sigmoid(logits_pair)
        pair_target = tf.cast(pair_target, tf.float32)
        per_example_contrast_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_pair , labels=pair_target),
                                         axis=-1)
        contrast_loss = tf.reduce_mean(per_example_contrast_loss )


    return loss+contrast_loss, per_example_loss, probabilities, predict_ids, labels



def model_fn_builder(bert_config, num_token_labels, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        token_label_ids = features["token_label_ids"]
        label_ids = features["label_ids"]
        fit_labelspace_positions = features["fit_labelspace_positions"]
        fit_docspace_positions = features["fit_docspace_positions"]
        pair = features["pair"]
        pair_target = features["pair_target"]
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, token_label_per_example_loss, token_label_logits, token_label_predictions, token_label_ids_labelspace) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, token_label_ids, label_ids, fit_labelspace_positions, fit_docspace_positions,
            num_token_labels, num_labels, use_one_hot_embeddings, pair, pair_target)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        export_outputs = {
            'predict_output': tf.estimator.export.PredictOutput({
                'token_label_logits': token_label_logits,
                'token_label_predictions': token_label_predictions
            })
        }
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,
            FLAGS.layerwise_lr_decay, bert_config.num_hidden_layers) #num_hidden_layers=12
            logging_hook = tf.train.LoggingTensorHook({"total_loss": total_loss}, every_n_iter=1000)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook],
                export_outputs=export_outputs,
                scaffold=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(token_label_predictions, num_labels, token_label_ids, positions, token_label_ids_labelspace, token_label_per_example_loss, is_real_example):
                token_label_ids = token_label_ids_labelspace
                pos_indices_list = list(range(num_token_labels))[1:2]  #0/1
                token_label_precision_micro = tf_metrics.precision(token_label_ids, token_label_predictions,
                                                                   num_token_labels,
                                                                   pos_indices_list, average="micro")
                token_label_recall_micro = tf_metrics.recall(token_label_ids, token_label_predictions, num_token_labels,
                                                             pos_indices_list, average="micro")
                token_label_f_micro = tf_metrics.f1(token_label_ids, token_label_predictions, num_token_labels,
                                                    pos_indices_list,
                                                    average="micro")
                token_label_loss = tf.metrics.mean(values=token_label_per_example_loss, weights=is_real_example)

                #HammingLoss
                aa = tf.cast(token_label_ids, tf.int32)
                bb = tf.cast(token_label_predictions, tf.int32)
                no_elements_equal = tf.cast(tf.not_equal(aa, bb), tf.int32)
                row_predict_ids = tf.cast(tf.reduce_sum(no_elements_equal, axis=-1), tf.float32)
                row_label_ids = tf.cast(tf.reduce_sum(tf.ones_like(token_label_ids), axis=-1), tf.float32)
                per_instance = tf.divide(row_predict_ids, row_label_ids)
                hamming_loss = tf.metrics.mean(values=per_instance)
                return {
                    "eval_token_label_precision(micro)": token_label_precision_micro,
                    "eval_token_label_recall(micro)": token_label_recall_micro,
                    "eval_token_label_f(micro)": token_label_f_micro,
                    "eval_token_label_loss": token_label_loss,
                    "eval_hamming_loss":hamming_loss
                }

            eval_metrics = metric_fn(token_label_predictions, num_labels, token_label_ids, fit_labelspace_positions, token_label_ids_labelspace, token_label_per_example_loss, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics,
                export_outputs=export_outputs,
                scaffold=scaffold_fn)
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={'token_label_logits': token_label_logits,
                             'token_label_predictions': token_label_predictions,
                             },
                export_outputs=export_outputs,
                scaffold=scaffold_fn)
        return output_spec

    return model_fn


def run_pred():
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "laco_plcp": Multi_Label_Classification_Processor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    token_label_list = processor.get_token_labels()
    label_list = processor.get_labels()

    num_token_labels = len(token_label_list)
    label_length = len(label_list)

    token_label_id2label = {}
    for (i, label) in enumerate(token_label_list):
        token_label_id2label[i] = label

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    run_config = tf.estimator.RunConfig(
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=2000,
        keep_checkpoint_max=50,
    )

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_token_labels=num_token_labels,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config,
        params={
            'batch_size': FLAGS.train_batch_size
        })

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            train_examples, token_label_list, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        #change
        early_stopping_hook = tf.estimator.experimental.stop_if_no_increase_hook(estimator=estimator,
                                                                       metric_name='eval_token_label_f(micro)',
                                                                       max_steps_without_increase=60000,
                                                                       run_every_secs=None,
                                                                       run_every_steps=2000)

        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            label_length=label_length,
            is_training=True,
            drop_remainder=True)
        if FLAGS.train_and_evaluate:
            train_spec = tf.estimator.TrainSpec(
                input_fn=train_input_fn,
                max_steps=num_train_steps,
                hooks=[early_stopping_hook])

            eval_examples = processor.get_dev_examples(FLAGS.data_dir)
            eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
            file_based_convert_examples_to_features(
                eval_examples, token_label_list, label_list, FLAGS.max_seq_length, tokenizer, eval_file)
            eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=FLAGS.max_seq_length,
                label_length=label_length,
                is_training=False,
                drop_remainder=False)
            eval_spec = tf.estimator.EvalSpec(
                input_fn=eval_input_fn)

            tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        else:
            estimator.train(
                input_fn=train_input_fn,
                max_steps=num_train_steps,
                hooks=[early_stopping_hook])

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, token_label_list, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            label_length=label_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("%s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        start = time.clock()
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, token_label_list, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            label_length=label_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_score_value_file = os.path.join(FLAGS.output_dir, "predicate_score_value.txt")
        output_predicate_predict_file = os.path.join(FLAGS.output_dir, "predicate_predict.txt")
        with tf.gfile.GFile(output_score_value_file, "w") as score_value_writer:
            with tf.gfile.GFile(output_predicate_predict_file, "w") as predicate_predict_writer:
                num_written_lines = 0
                tf.logging.info("***** Predict results *****")
                for (i, prediction) in enumerate(result):
                    token_label_prediction = prediction["token_label_predictions"]
                    token_label_logits = prediction["token_label_logits"]
                    if i >= num_actual_predict_examples:
                        break
                    token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction) + "\n"
                    predicate_predict_writer.write(token_label_output_line)
                    predicate_probabilities_line = " ".join(
                        str(sigmoid_logit) for sigmoid_logit in token_label_logits) + "\n"
                    score_value_writer.write(predicate_probabilities_line)
                    num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples

        if FLAGS.do_test_with_results:
            tf.logging.info("**** evaluate main ****")
            evaluate_main(FLAGS.init_path_dir, job_id, FLAGS.model_idx_info)

        elapsed = (time.clock() - start)
        tf.logging.info("  time_use  = %f", elapsed)



def predicate_classification_model_train(job_id, schema_set, train_data_path, train_para):
    ie_save_path_re = train_data_path
    path = os.path.join(train_data_path, "temp_data/predicate_classifiction/classification_data")
    FLAGS.task_name = "LACO_plcp"
    FLAGS.do_train = True
    FLAGS.do_eval = False
    FLAGS.train_and_evaluate = True
    FLAGS.data_dir = path
    pretrain_model_path = get_pretrain_model_path()
    path = os.path.join(pretrain_model_path, "cased_L-12_H-768_A-12/vocab.txt")
    FLAGS.vocab_file = path
    path = os.path.join(pretrain_model_path, "cased_L-12_H-768_A-12/bert_config.json")
    FLAGS.bert_config_file = path
    path = os.path.join(pretrain_model_path, "cased_L-12_H-768_A-12/bert_model.ckpt")
    FLAGS.init_checkpoint = path
    FLAGS.max_seq_length = train_para['para_max_len']
    FLAGS.train_batch_size = train_para['para_batch_size']
    FLAGS.learning_rate = train_para['para_learning_rate']
    FLAGS.num_train_epochs = train_para['para_epochs']
    label = schema_set.split("#")
    FLAGS.labels_list = label
    path = os.path.join(ie_save_path_re, "model/predicate_infer_out_plcp_train/")
    FLAGS.output_dir = path
    run_pred()

def predicate_classification_model_test(job_id, schema_set, train_data_path, clf_model_idx):
    ie_save_path_re = get_ie_config("re")
    FLAGS.init_path_dir = ie_save_path_re
    path_re = os.path.join(train_data_path, "log", job_id)
    path = os.path.join(path_re, "temp_data/predicate_classifiction/classification_data")
    FLAGS.task_name = "LACO_plcp"
    FLAGS.do_predict = True
    FLAGS.do_test_with_results = True
    FLAGS.data_dir = path
    pretrain_model_path = get_pretrain_model_path()
    path = os.path.join(pretrain_model_path, "cased_L-12_H-768_A-12/vocab.txt")
    FLAGS.vocab_file = path
    path = os.path.join(pretrain_model_path, "cased_L-12_H-768_A-12/bert_config.json")
    FLAGS.bert_config_file = path
    path = os.path.join(ie_save_path_re, "model/predicate_infer_out_plcp_best/"+clf_model_idx)
    FLAGS.model_idx_info = clf_model_idx
    FLAGS.init_checkpoint = path
    train_para = get_train_para()
    FLAGS.max_seq_length = train_para['para_max_len']
    path = os.path.join(path_re, "temp_data/predicate_infer_out/")
    FLAGS.output_dir = path
    label = schema_set.split("#")
    FLAGS.labels_list = label
    run_pred()

if __name__ == '__main__':
    job_id = get_job_id()
    train_data_path = get_ie_config("re")
    schema_set = get_schema_set()
    train_para = get_train_para()
    predicate_classification_model_train(job_id, schema_set, train_data_path, train_para)
