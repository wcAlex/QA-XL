import argparse
import os

def get_setup_args():
    """Get arguments needed in setup.py."""
    parser = argparse.ArgumentParser('Download and pre-process SQuAD')

    add_common_args(parser)

    parser.add_argument('--train_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/train-v2.0.json')
    parser.add_argument('--dev_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/dev-v2.0.json')
    parser.add_argument('--test_url',
                        type=str,
                        default='https://github.com/chrischute/squad/data/test-v2.0.json')
    parser.add_argument('--glove_url',
                        type=str,
                        default='http://nlp.stanford.edu/data/glove.840B.300d.zip')
    parser.add_argument('--dev_meta_file',
                        type=str,
                        default='./data/dev_meta.json')
    parser.add_argument('--test_meta_file',
                        type=str,
                        default='./data/test_meta.json')
    parser.add_argument('--word2idx_file',
                        type=str,
                        default='./data/word2idx.json')
    parser.add_argument('--char2idx_file',
                        type=str,
                        default='./data/char2idx.json')
    parser.add_argument('--answer_file',
                        type=str,
                        default='./data/answer.json')
    parser.add_argument('--para_limit',
                        type=int,
                        default=400,
                        help='Max number of words in a paragraph')
    parser.add_argument('--ques_limit',
                        type=int,
                        default=50,
                        help='Max number of words to keep from a question')
    parser.add_argument('--test_para_limit',
                        type=int,
                        default=1000,
                        help='Max number of words in a paragraph at test time')
    parser.add_argument('--test_ques_limit',
                        type=int,
                        default=100,
                        help='Max number of words in a question at test time')
    parser.add_argument('--char_dim',
                        type=int,
                        default=64,
                        help='Size of char vectors (char-level embeddings)')
    parser.add_argument('--glove_dim',
                        type=int,
                        default=300,
                        help='Size of GloVe word vectors to use')
    parser.add_argument('--glove_num_vecs',
                        type=int,
                        default=2196017,
                        help='Number of GloVe vectors')
    parser.add_argument('--ans_limit',
                        type=int,
                        default=30,
                        help='Max number of words in a training example answer')
    parser.add_argument('--char_limit',
                        type=int,
                        default=16,
                        help='Max number of chars to keep from a word')
    parser.add_argument('--include_test_examples',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Process examples from the test set')

    args = parser.parse_args()

    return args

def add_common_args(parser):
    """Add arguments common to all 3 scripts: setup.py, train.py, test.py"""
    parser.add_argument('--train_record_file',
                        type=str,
                        default='./data/train.npz')
    parser.add_argument('--dev_record_file',
                        type=str,
                        default='./data/dev.npz')
    parser.add_argument('--test_record_file',
                        type=str,
                        default='./data/test.npz')
    parser.add_argument('--word_emb_file',
                        type=str,
                        default='./data/word_emb.json')
    parser.add_argument('--char_emb_file',
                        type=str,
                        default='./data/char_emb.json')
    parser.add_argument('--train_eval_file',
                        type=str,
                        default='./data/train_eval.json')
    parser.add_argument('--dev_eval_file',
                        type=str,
                        default='./data/dev_eval.json')
    parser.add_argument('--test_eval_file',
                        type=str,
                        default='./data/test_eval.json')


def setup_args():
    """Get arguments needed in train.py."""
    parser = argparse.ArgumentParser('Train/Test a model on SQuAD')
    
    ### parameters ###
    parser.add_argument('--data_dir', default='./SQuAD/')
    parser.add_argument('--model_dir', default='train/best_model')
    parser.add_argument('--answer_file', default='test/' + 'train/best_model'.split('/')[-1] + '.answers')
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--seed', default=1234)
    parser.add_argument('--epochs', type = int, default=20)
    parser.add_argument('--eval', type = bool, default=False)
    parser.add_argument('--load_model', type = bool, default = False)
    parser.add_argument('--batch_size', type = int, default=12)
    parser.add_argument('--grad_clipping', type = float, default = 10)
    parser.add_argument('--lrate', type = float, default=0.002)
    parser.add_argument('--dropout', type = float, default=0.3)
    parser.add_argument('--use_xl', type = bool, default=True)
    parser.add_argument('--fix_embeddings', type = bool, default=False)
    parser.add_argument('--char_dim', type = int, default=64)
    parser.add_argument('--pos_dim', type = int, default=12)
    parser.add_argument('--ner_dim', type = int, default=8)
    parser.add_argument('--char_hidden_size', type = int, default=50)
    parser.add_argument('--hidden_size', type = int, default=128)
    parser.add_argument('--attention_size', type = int, default=250)
    parser.add_argument('--decay_period', type = int, default=10)
    parser.add_argument('--decay', type = int, default=0.5)
    args = parser.parse_args()

    if not os.path.exists('train/'): ##train_model/
        os.makedirs('train/')
    if not os.path.exists('test/'): ##result/
        os.makedirs('test/')

    return args
