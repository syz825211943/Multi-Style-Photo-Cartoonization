from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = True

        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--init', action='store_true', help='whether use initialization phase ')
        
        self.parser.add_argument('--encoder_test', action='store_true', help='test the function of encoder part')
        self.parser.add_argument('--which_epoch', type=int, default=0, help='which epoch to load if continuing training')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc (determines name of folder to load from)')

        self.parser.add_argument('--init_niter', type=int, default=20, help='# of epochs at starting learning rate (try 50*n_domains)')
        self.parser.add_argument('--niter', type=int, default=100, help='# of epochs at starting learning rate (including initialization phase)')
        self.parser.add_argument('--niter_decay', type=int, default=50, help='# of epochs to linearly decay learning rate to zero (try 50*n_domains)')

        self.parser.add_argument('--init_lr', type=float, default=0.0002, help='initial learning rate for ADAM') # initial 0.0002
        self.parser.add_argument('--lr', type=float, default=0.00001, help='learning rate for ADAM') # initial 0.0002
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of ADAM')

        self.parser.add_argument('--lambda_content', type=float, default=0.2, help='weight for content loss between real images and generated images')
        self.parser.add_argument('--lambda_classfication', type=float, default=0.5, help='weight for forward loss (A -> B; try 0.2)')

        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        self.parser.add_argument('--no_lsgan', action='store_true', help='use vanilla discriminator in place of least-squares one')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
