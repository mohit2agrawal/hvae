import argparse
import os


def float_limited(x):
    '''only accept float values in range [0,1]'''
    x = float(x)
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError('float value in range [0,1] expected')
    return x


class Parameters():
    # general parameters
    num_topics = 10
    ntm_hidden = 256
    anneal_value = 1
    debug = True
    #name = "yelp"
    # name = "amazon"
    name = "imdb_topic"
    num_samples = 200
    # std=13, inputless_dec(dec_keep_rate=0.0)=111------------------------------>
    latent_size = 150
    num_epochs = 50  ## for ours epoch = iters
    learning_rate = 0.0005
    batch_size = 16
    encoder_hidden = 600  # std=191, inputless_dec=350
    decoder_hidden = 600  # ----------------------------------------------------->modify param
    # for decoding
    temperature = 1.0
    gen_length = 40
    # beam search
    beam_search = False
    beam_size = 2
    # encoder
    rnn_layers = 1
    encode = 'hw'  # 'hw' or 'mlp'
    # highway networks
    keep_rate = 1.0  # --------------------------------------------------->
    highway_lc = 1  # ------------------------------------------------->
    highway_ls = 600
    # decoder
    decoder_rnn_layers = 1
    dec_keep_rate = 0.62
    decode = 'hw'  # can use 'hw', 'concat', 'mlp'
    # data
    datasets = ['GOT', 'PTB']
    embed_size = 300  # std=353, inputless_dec=499
    label_embed_size = 16
    sent_max_size = 1000
    input_ = datasets[1]
    debug = False
    vocab_drop = 4  # drop less than n times occured
    # use pretrained w2vec embeddings
    pre_trained_embed = True
    fine_tune_embed = False
    # technical parameters
    is_training = True
    LOG_DIR = './model_logs_' + name + "_" + str(num_topics) + "/"
    MODEL_DIR = './models_ckpts_' + name + "_" + str(num_topics) + "/"
    visualise = False
    # gru base cell partially implemented
    base_cell = 'lstm'  # or GRU

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Specify some parameters, all parameters "
            "also can be directly specified in Parameters class"
        )
        parser.add_argument('--name', default=self.name)
        parser.add_argument(
            '--dataset',
            default=self.input_,
            help='training dataset (GOT or PTB)',
            dest='data'
        )
        parser.add_argument(
            '--lr', default=self.learning_rate, help='learning rate', dest='lr'
        )
        parser.add_argument(
            '--embed_dim',
            default=self.embed_size,
            help='embedding size',
            dest='embed'
        )
        parser.add_argument(
            '--lst_state_dim_enc',
            default=self.encoder_hidden,
            help='encoder state size',
            dest='enc_hid'
        )
        parser.add_argument(
            '--lst_state_dim_dec',
            default=self.decoder_hidden,
            help='decoder state size',
            dest='dec_hid'
        )
        parser.add_argument(
            '--latent',
            default=self.latent_size,
            help='latent space size',
            dest='latent'
        )
        parser.add_argument(
            '--dec_dropout',
            default=self.dec_keep_rate,
            help='decoder dropout keep rate',
            dest='dec_drop'
        )
        parser.add_argument(
            '--beam_search', default=self.beam_search, action="store_true"
        )
        parser.add_argument('--beam_size', default=self.beam_size)
        parser.add_argument(
            '--decode',
            default=self.decode,
            help='define mapping from z->lstm. mlp, concat, hw'
        )
        parser.add_argument(
            '--encode',
            default=self.encode,
            help='define mapping from lstm->z. mlp, hw'
        )
        parser.add_argument(
            '--vocab_drop', default=self.vocab_drop, help='drop less than'
        )
        parser.add_argument('--gpu', default="0", help="specify GPU number")

        parser.add_argument(
            '--cycles', default=1, type=int, help="number of cycles"
        )
        parser.add_argument(
            '--cycle_proportion',
            default=0.5,
            type=float_limited,
            help="proportion of cycle used to increase alpha and beta"
        )
        parser.add_argument(
            '--fn',
            default="linear",
            choices=["linear", "tanh", "cosine"],
            help="function used for increasing alpha"
        )
        parser.add_argument(
            '--beta_lag',
            default=0,
            type=float,
            help="proportion of cycle beta lags behind alpha"
        )
        parser.add_argument(
            '--zero_start',
            default=0,
            type=float_limited,
            help="proportion of cycle used to increase alpha and beta"
        )

        parser.add_argument(
            '--ckpt_path',
            help="path to the checkpoint to load (for sampling/ generation)"
        )
        parser.add_argument(
            '--num_samples',
            type=int,
            default=self.num_samples,
            help="number of sentences to generate"
        )
        parser.add_argument(
            '--num_topics',
            type=int,
            default=self.num_topics,
            help="number of topics"
        )

        args = parser.parse_args()
        self.input_ = args.data
        self.learning_rate = float(args.lr)
        self.embed_size = int(args.embed)
        self.encoder_hidden = int(args.enc_hid)
        self.decoder_hidden = int(args.dec_hid)
        self.latent_size = int(args.latent)
        self.dec_keep_rate = float(args.dec_drop)
        self.beam_search = args.beam_search
        self.beam_size = int(args.beam_size)
        self.decode = args.decode
        self.encode = args.encode
        self.vocab_drop = int(args.vocab_drop)

        self.cycles = args.cycles
        self.cycle_proportion = args.cycle_proportion
        self.fn = args.fn
        self.beta_lag = args.beta_lag
        self.zero_start = args.zero_start

        self.ckpt_path = args.ckpt_path
        self.num_samples = args.num_samples

        self.name = args.name
        self.num_topics = args.num_topics

        # uncomment to make it GPU
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def test():
    params = Parameters()
    params.parse_args()
    print(type(params.cycles), params.cycles)
    print(type(params.cycle_proportion), params.cycle_proportion)
    print(type(params.fn), params.fn)
    print(type(params.beta_lag), params.beta_lag)


if __name__ == '__main__':
    test()
