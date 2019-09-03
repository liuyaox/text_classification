# coding:utf-8
import sys
import os
import tempfile
import random
import json
import logging
from termcolor import colored
import contextlib
from queue import Queue
from threading import Thread
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from model.Bert import modeling
from model.Bert import tokenization
from model.Bert import args
from model.Bert.args import PoolingStrategy


def import_tf(device_id=-1, verbose=False):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if device_id < 0 else str(device_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' if verbose else '3'    
    tf.logging.set_verbosity(tf.logging.DEBUG if verbose else tf.logging.ERROR)
    return tf

tf = import_tf(0, True)


def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).5s:%(funcName).3s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger
    

class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

        
def optimize_graph(config_name,
                   ckpt_name,
                   logger=None, 
                   verbose=False, 
                   pooling_strategy=PoolingStrategy.REDUCE_MEAN, 
                   max_seq_len=40,
                   graph_tmpfile="./tmpxxx"):
    if not logger:
        logger = set_logger(colored('BERT_VEC', 'yellow'), verbose)
    try:
        # we don't need GPU for optimizing the graph
        tf = import_tf(device_id=0, verbose=verbose)
        from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

        # allow_soft_placement:自动选择运行设备
        config = tf.ConfigProto(allow_soft_placement=True)
        config_fp = config_name
        init_checkpoint = ckpt_name
        logger.info('model config: %s' % config_fp)

        # 加载bert配置文件
        with tf.gfile.GFile(config_fp, 'r') as f:
            bert_config = modeling.BertConfig.from_dict(json.load(f))

        logger.info('build graph...')
        # input placeholders, not sure if they are friendly to XLA
        input_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_ids')
        input_mask = tf.placeholder(tf.int32, (None, max_seq_len), 'input_mask')
        input_type_ids = tf.placeholder(tf.int32, (None, max_seq_len), 'input_type_ids')

        # xla加速
        jit_scope = tf.contrib.compiler.jit.experimental_jit_scope if args.xla else contextlib.suppress

        with jit_scope():
            input_tensors = [input_ids, input_mask, input_type_ids]

            model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids,
                use_one_hot_embeddings=False)

            # 获取所有要训练的变量
            tvars = tf.trainable_variables()

            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)

            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

            minus_mask = lambda x, m: x - tf.expand_dims(1.0 - m, axis=-1) * 1e30
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_max = lambda x, m: tf.reduce_max(minus_mask(x, m), axis=1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            # 共享卷积核
            with tf.variable_scope("pooling"):
                # 如果只有一层，就只取对应那一层的weight
                if len(args.layer_indexes) == 1:
                    encoder_layer = model.all_encoder_layers[args.layer_indexes[0]]
                else:
                    # 否则遍历需要取的层，把所有层的weight取出来并拼接起来shape:768*层数
                    all_layers = [model.all_encoder_layers[l] for l in args.layer_indexes]
                    encoder_layer = tf.concat(all_layers, -1)

                input_mask = tf.cast(input_mask, tf.float32)

                # 以下代码是句向量的生成方法，可以理解为做了一个卷积的操作，但是没有把结果相加, 卷积核是input_mask
                if pooling_strategy == PoolingStrategy.REDUCE_MEAN:
                    pooled = masked_reduce_mean(encoder_layer, input_mask)
                elif pooling_strategy == PoolingStrategy.REDUCE_MAX:
                    pooled = masked_reduce_max(encoder_layer, input_mask)
                elif pooling_strategy == PoolingStrategy.REDUCE_MEAN_MAX:
                    pooled = tf.concat([masked_reduce_mean(encoder_layer, input_mask),
                                        masked_reduce_max(encoder_layer, input_mask)], axis=1)
                elif pooling_strategy == PoolingStrategy.FIRST_TOKEN or \
                        pooling_strategy == PoolingStrategy.CLS_TOKEN:
                    pooled = tf.squeeze(encoder_layer[:, 0:1, :], axis=1)
                elif pooling_strategy == PoolingStrategy.LAST_TOKEN or \
                        pooling_strategy == PoolingStrategy.SEP_TOKEN:
                    seq_len = tf.cast(tf.reduce_sum(input_mask, axis=1), tf.int32)
                    rng = tf.range(0, tf.shape(seq_len)[0])
                    indexes = tf.stack([rng, seq_len - 1], 1)
                    pooled = tf.gather_nd(encoder_layer, indexes)
                elif pooling_strategy == PoolingStrategy.NONE:
                    pooled = mul_mask(encoder_layer, input_mask)
                else:
                    raise NotImplementedError()

            pooled = tf.identity(pooled, 'final_encodes')

            output_tensors = [pooled]
            tmp_g = tf.get_default_graph().as_graph_def()

        with tf.Session(config=config) as sess:
            logger.info('load parameters from checkpoint...')
            sess.run(tf.global_variables_initializer())
            logger.info('freeze...') 
            tmp_g = tf.graph_util.convert_variables_to_constants(sess, tmp_g, [n.name[:-2] for n in output_tensors])
            dtypes = [n.dtype for n in input_tensors]
            logger.info('optimize...')
            tmp_g = optimize_for_inference(
                tmp_g,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                [dtype.as_datatype_enum for dtype in dtypes],
                False)
        logger.info('write graph to a tmp file: %s' % graph_tmpfile)
        with tf.gfile.GFile(graph_tmpfile, 'wb') as f:
            f.write(tmp_g.SerializeToString())
        return graph_tmpfile
    except Exception as e:
        logger.error('fail to optimize the graph!')
        logger.error(e)


class BertVector:
    def __init__(self, batch_size=1, 
                       pooling_strategy="REDUCE_MEAN", 
                       max_seq_len=40,
                       bert_model_path="./chinese_L-12_H-768_A-12/",
                       graph_tmpfile="./tmpxxx"):
        """
        init BertVector
        :param batch_size:     Depending on your memory default is 32
        """
        self.max_seq_length = max_seq_len
        self.layer_indexes = args.layer_indexes
        self.gpu_memory_fraction = 1
        
        self.file_path = os.path.dirname(__file__)

        self.model_dir = os.path.join(self.file_path, bert_model_path)
        self.config_name = os.path.join(self.model_dir, 'bert_config.json')
        self.ckpt_name = os.path.join(self.model_dir, 'bert_model.ckpt')
        self.vocab_file = os.path.join(self.model_dir, 'vocab.txt')

        if pooling_strategy == "NONE":
            pooling_strategy = PoolingStrategy.NONE
        elif pooling_strategy == "REDUCE_MAX":
            pooling_strategy = PoolingStrategy.REDUCE_MAX
        elif pooling_strategy == "REDUCE_MEAN":
            pooling_strategy = PoolingStrategy.REDUCE_MEAN
        elif pooling_strategy == "REDUCE_MEAN_MAX":
            pooling_strategy = PoolingStrategy.REDUCE_MEAN_MAX

        self.graph_path = optimize_graph(self.config_name,
                                         self.ckpt_name,
                                         pooling_strategy=pooling_strategy, 
                                         max_seq_len=self.max_seq_length, 
                                         graph_tmpfile=graph_tmpfile)

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)
        self.batch_size = batch_size
        self.estimator = self.get_estimator()
        self.input_queue = Queue(maxsize=1)
        self.output_queue = Queue(maxsize=1)
        self.predict_thread = Thread(target=self.predict_from_queue, daemon=True)
        self.predict_thread.start()

    def get_estimator(self):
        from tensorflow.python.estimator.estimator import Estimator
        from tensorflow.python.estimator.run_config import RunConfig
        from tensorflow.python.estimator.model_fn import EstimatorSpec

        def model_fn(features, labels, mode, params):
            with tf.gfile.GFile(self.graph_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            input_names = ['input_ids', 'input_mask', 'input_type_ids']

            output = tf.import_graph_def(graph_def,
                                         input_map={k + ':0': features[k] for k in input_names},
                                         return_elements=['final_encodes:0'])

            return EstimatorSpec(mode=mode, predictions={
                'encodes': output[0]
            })

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        return Estimator(model_fn=model_fn, config=RunConfig(session_config=config),
                         params={'batch_size': self.batch_size})

    def predict_from_queue(self):
        prediction = self.estimator.predict(input_fn=self.queue_predict_input_fn, yield_single_examples=False)
        for i in prediction:
            self.output_queue.put(i)

    def encode(self, sentence):
        self.input_queue.put(sentence)
        prediction = self.output_queue.get()
        return prediction

    def queue_predict_input_fn(self):

        return (tf.data.Dataset.from_generator(
            self.generate_from_queue,
            output_types={'unique_ids': tf.int32,
                          'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32},
            output_shapes={
                'unique_ids': (1,),
                'input_ids': (None, self.max_seq_length),
                'input_mask': (None, self.max_seq_length),
                'input_type_ids': (None, self.max_seq_length)}))

    def generate_from_queue(self):
        while True:
            features = list(self.convert_examples_to_features(seq_length=self.max_seq_length, tokenizer=self.tokenizer))
            yield {
                'unique_ids': [f.unique_id for f in features],
                'input_ids': [f.input_ids for f in features],
                'input_mask': [f.input_mask for f in features],
                'input_type_ids': [f.input_type_ids for f in features]
            }

    def input_fn_builder(self, features, seq_length):
        """Creates an `input_fn` closure to be passed to Estimator."""

        all_unique_ids = []
        all_input_ids = []
        all_input_mask = []
        all_input_type_ids = []

        for feature in features:
            all_unique_ids.append(feature.unique_id)
            all_input_ids.append(feature.input_ids)
            all_input_mask.append(feature.input_mask)
            all_input_type_ids.append(feature.input_type_ids)

        def input_fn(params):
            """The actual input function."""
            batch_size = params["batch_size"]

            num_examples = len(features)

            # This is for demo purposes and does NOT scale to large data sets. We do
            # not use Dataset.from_generator() because that uses tf.py_func which is
            # not TPU compatible. The right way to load data is with TFRecordReader.
            d = tf.data.Dataset.from_tensor_slices({
                "unique_ids":
                    tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_type_ids":
                    tf.constant(
                        all_input_type_ids,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
            })

            d = d.batch(batch_size=batch_size, drop_remainder=False)
            return d

        return input_fn

    def model_fn_builder(self, bert_config, init_checkpoint, layer_indexes):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            unique_ids = features["unique_ids"]
            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            input_type_ids = features["input_type_ids"]

            jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

            with jit_scope():
                model = modeling.BertModel(
                    config=bert_config,
                    is_training=False,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    token_type_ids=input_type_ids)

                if mode != tf.estimator.ModeKeys.PREDICT:
                    raise ValueError("Only PREDICT modes are supported: %s" % (mode))

                tvars = tf.trainable_variables()

                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                           init_checkpoint)

                tf.logging.info("**** Trainable Variables ****")
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)

                all_layers = model.get_all_encoder_layers()

                predictions = {
                    "unique_id": unique_ids,
                }

                for (i, layer_index) in enumerate(layer_indexes):
                    predictions["layer_output_%d" % i] = all_layers[layer_index]

                from tensorflow.python.estimator.model_fn import EstimatorSpec

                output_spec = EstimatorSpec(mode=mode, predictions=predictions)
                return output_spec

        return model_fn

    def convert_examples_to_features(self, seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        features = []
        input_masks = []
        examples = self._to_example(self.input_queue.get())
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            # if the sentences's length is more than seq_length, only use sentence's left part
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

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
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            input_type_ids = []
            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            # Where "input_ids" are tokens's index in vocabulary
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)
            input_masks.append(input_mask)
            # Zero-pad up to the sequence length.
            while len(input_ids) < seq_length:
                input_ids.append(0)
                input_mask.append(0)
                input_type_ids.append(0)

            assert len(input_ids) == seq_length
            assert len(input_mask) == seq_length
            assert len(input_type_ids) == seq_length

            if ex_index < 5:
                tf.logging.info("*** Example ***")
                tf.logging.info("unique_id: %s" % (example.unique_id))
                tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
                tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                tf.logging.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

            yield InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids)

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    @staticmethod
    def _to_example(sentences):
        import re
        """
        sentences to InputExample
        :param sentences: list of strings
        :return: list of InputExample
        """
        unique_id = 0
        for ss in sentences:
            line = tokenization.convert_to_unicode(ss)
            if not line:
                continue
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            yield InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b)
            unique_id += 1


if __name__ == "__main__":
    bc = BertVector(batch_size=1, 
                      pooling_strategy="REDUCE_MEAN", 
                      max_seq_len=20,
                      bert_model_path="D:\\workspaces\\code\\tfhub\\keras_dssm\\chinese_L-12_H-768_A-12\\",
                      graph_tmpfile="D:\\workspaces\\code\\tfhub\\keras_dssm\\tmpxxx")
    query = u"新浪移动"
    vectors = bc.encode([query])
    print(str(vectors))
    