import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter
import functools, itertools
import numpy as np
from torch.autograd import Variable

def define_G(framework, input_nc, output_nc, ngf, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d
    n_blocks -= n_blocks_shared
    n_blocks_enc = 5
    n_blocks_dec = 4

    dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if n_blocks_shared > 0:
        n_blocks_shdec = n_blocks_shared // 2
        n_blocks_shenc = n_blocks_shared - n_blocks_shdec
        shenc_args = (n_domains, n_blocks_shenc) + dup_args
        shdec_args = (n_domains, n_blocks_shdec) + dup_args
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args, ResnetGenShared, shenc_args, shdec_args)
    else:
        if(framework == 'cartoon_generator'):
          plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder_upsample, dec_args)
        elif(framework == 'base'):
          plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args)
        elif(framework == 'ResnetGenDecoder_AdaIN_upblock'):
          plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder_AdaIN_upblock, dec_args)
        elif(framework == 'ResnetGenDecoder_AdaIN_decoder'):
          plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder_AdaIN_decoder, dec_args)
        elif(framework == 'ResnetGenDecoder_ILN_upblock'):
          plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder_ILN_upblock, dec_args)
        elif(framework == 'ResnetGenDecoder_ILN_decoder'):
          plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder_ILN_decoder, dec_args)
        elif(framework == 'ResnetGenEncoder_gram'):
          plex_netG = G_Plexer(n_domains, ResnetGenEncoder_gram, enc_args, ResnetGenDecoder, dec_args)
        elif(framework == 'ResnetGenEncoder_onehot_input'):
          plex_netG = G_Plexer(n_domains, ResnetGenEncoder_onehot_input, enc_args, ResnetGenDecoder, dec_args)

    # set the GPU
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    # apply the initialization of weights
    plex_netG.apply(weights_init)
    return plex_netG


def define_D(framework, input_nc, ndf, netD_n_layers, n_domains, norm='batch', gpu_ids=[]):
    # set the instance normalization/batch normalization
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, norm_layer, gpu_ids)
    # choose discriminator framework here
    if(framework == 'cartoon_discriminator'):
        plex_netD = D_Plexer(n_domains, NLayerDiscriminator, model_args)
    elif(framework == 'Discriminator_class'):
        plex_netD = D_Plexer(n_domains, Discriminator_class, model_args)

    # set the GPU
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])
    # initialization
    plex_netD.apply(weights_init)
    return plex_netD

def define_VGG(init_weights_, feature_mode_, batch_norm_=False, num_classes_=1000, gpu_id_=[]):
    VGG = VGG19(init_weights=init_weights_, feature_mode=feature_mode_, batch_norm=batch_norm_, num_classes=num_classes_, gpu_id = gpu_id_)
    # set the GPU
    if len(gpu_id_) > 0:
        assert(torch.cuda.is_available())
        VGG.cuda(gpu_id_[0])

    if not init_weights_ == None:
      VGG.load_state_dict(torch.load(init_weights_))
      print('load VGG weights successfully')
    return VGG

# defin aux classifier
def define_classifier(framework='basic', gpu_ids = []):
    if framework == 'basic':
        aux_classifier_ = aux_classifier(gpu_id = gpu_ids)
    elif framework == 'conv4': # conv classifier for 4 class
        aux_classifier_ = aux_classifier_conv_4(gpu_id = gpu_ids)
      # set the GPU
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        aux_classifier_.cuda(gpu_ids[0])
    # apply the initialization of weights
    aux_classifier_.apply(weights_init)
    return aux_classifier_


# weights initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.Tensor = tensor
        self.labels_real, self.labels_fake = None, None
        self.preloss = nn.Sigmoid() if not use_lsgan else None
        self.loss = nn.MSELoss() if use_lsgan else nn.BCELossGANLoss()

    def get_target_tensor(self, inputs, is_real):
        self.labels_real = self.Tensor(inputs.size()).fill_(1.0) 
        self.labels_fake = self.Tensor(inputs.size()).fill_(0.0)
        if is_real:
          return self.labels_real
        return self.labels_fake

    def __call__(self, inputs, is_real):
        inputs_ = inputs.clone()
        labels = self.get_target_tensor(inputs_, is_real)

        return self.loss(inputs,labels)

class ClassLoss(nn.Module):
    def __init__(self, tensor=torch.FloatTensor):
        super(ClassLoss, self).__init__()
        self.Tensor = tensor
        self.labels = None
        self.loss = nn.CrossEntropyLoss()

    def get_target_tensor(self, inputs, domain):
        self.labels = self.Tensor(inputs.size()[0]).fill_(domain)
        return self.labels

    def __call__(self, inputs, domain):
        inputs_ = inputs.clone()
        h = inputs_.size()[0]
        w = inputs_.size()[1]
        # inputs_ = torch.reshape(inputs_,(h,w))
        labels = self.get_target_tensor(inputs_, domain)
        labels = labels.long()
    
        return self.loss(inputs_,labels)

class aux_classifier_conv_4(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=4, gpu_id = []):
        super(aux_classifier_conv_4, self).__init__()
        conv_dim = 32
        repeat_num = 5
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(256 / np.power(2, repeat_num))
        layers.append(nn.Conv2d(curr_dim, num_classes, kernel_size=kernel_size, bias=False))
        self.main = nn.Sequential(*layers)

        
    def forward(self, x):
        h = self.main(x)
        return h.view(h.size(0), h.size(1))

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        params = self.main.parameters()
        self.optimizers.append( opt(params, lr=lr, betas=betas))

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def set_requires_grad(self, requires_grad=False):
        for param in self.main.parameters():
            param.requires_grad = requires_grad
    
    def zero_grads(self):
        self.optimizers[0].zero_grad()

    def step_grads(self):
        self.optimizers[0].step()

    def load(self, save_path):
        filename = save_path + ('%d.pth' % 0)
        self.main.load_state_dict(torch.load(filename))

    def save(self, save_path):
        filename = save_path + ('%d.pth' % 0)
        torch.save(self.main.cpu().state_dict(), filename)

# use vgg model
class aux_classifier(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=3, gpu_id = []):
        super(aux_classifier, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 16 * 16, num_classes)
        )

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        # print(x.size())
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            # print(x.size())
            x = self.classifier(x)
        return x

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        # for the only one encoder
        params = self.features.parameters()
        self.optimizers.append( opt(params, lr=lr, betas=betas))
        params_ = self.classifier.parameters()
        self.optimizers.append( opt(params_, lr=lr, betas=betas))

    def set_requires_grad(self, requires_grad=False):
        for param in self.features.parameters():
            param.requires_grad = requires_grad
        for param in self.classifier.parameters():
            param.requires_grad = requires_grad
    
    def zero_grads(self):
        self.optimizers[0].zero_grad()
        self.optimizers[1].zero_grad()

    def step_grads(self):
        self.optimizers[0].step()
        self.optimizers[1].step()


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out

class AdaIN(nn.Module):
    def __init__(self):
        super(AdaIN, self).__init__()

    def calc_mean_std(self, input, eps=1e-5):
        batch_size, channels = input.shape[:2]

        reshaped = input.view(batch_size, channels, -1) # Reshape channel wise
        mean = torch.mean(reshaped, dim = 2).view(batch_size, channels, 1, 1) # Calculat mean and reshape
        std = torch.sqrt(torch.var(reshaped, dim=2)+eps).view(batch_size, channels, 1, 1) # Calculate variance, add epsilon (avoid 0 division),
                                                                                    # calculate std and reshape
        return mean, std

    def forward(self, input, gamma, beta, flag):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        if flag:
            output = gamma*((input - in_mean) / (in_var)) + beta  # Normalise, then modify mean and std
        else:
            output = (input - in_mean)/ (in_var)
        return output

class GramMatrix(nn.Module):  # compute gram matrix 
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
	
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
          nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding = 1, bias=use_bias),	
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class ResnetGenEncoder_onehot_input(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder_onehot_input, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc+3, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
    
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
          nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding = 1, bias=use_bias),   
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input, label):
        self.labels = torch.Tensor(input.size()).fill_(0)
        self.labels[:,label,:,:] = 1
        input = torch.cat([input, self.labels.cuda()], dim=1)
        return self.model(input)

class ResnetGenEncoder_gram(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder_gram, self).__init__()
        self.gpu_ids = gpu_ids

       
        first_block = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
    
        # two downsampling
        n_downsampling = 2
        i = 0
        mult = 2**i
        second_block = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
            nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding = 1, bias=use_bias),   
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        i = 1
        mult = 2**i
        model = [nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
            nn.Conv2d(ngf * mult * 2, ngf * mult * 2, kernel_size=3, stride=1, padding = 1, bias=use_bias),   
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]
        
        self.gram_matrix = GramMatrix()
        self.first_block = nn.Sequential(*first_block)
        self.second_block = nn.Sequential(*second_block)
        self.model = nn.Sequential(*model)

    def forward(self, image, feature_map):
        gram_matrix = self.gram_matrix(feature_map)
        b,c,_ = gram_matrix.size()
        gram_matrix = torch.reshape(gram_matrix,(b,c,c,1))
        gram_matrix = torch.repeat_interleave(gram_matrix, repeats=c, dim=3)
       
        x = self.first_block(image)
        x = self.second_block(x)

        return self.model(torch.cat([x, gram_matrix], dim=1) )  # concat gram matrix and input image

class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)

class BasicResnetGenDecoder(nn.Module):
    # basic generator without any addition
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
	
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        model += [nn.ConvTranspose2d(ngf * mult, ngf * mult / 2,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(ngf * mult / 2, ngf * mult / 2, kernel_size=3, stride=1, padding=1),
                      norm_layer(ngf * mult / 2),
                      nn.ReLU(True)]

	# second upsampling
	# k3n64s1/2 + k3n64s1
        model += [nn.ConvTranspose2d(ngf * mult/2, ngf * mult / 4,
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
		      nn.Conv2d(ngf * mult / 4, ngf * mult / 4, kernel_size=3, stride=1, padding=1),
                      norm_layer(ngf * mult / 4),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf * mult / 4, output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)


class ResnetGenDecoder_upsample(nn.Module):
    # ResnetGenDecoder_6_18_2
    # encoder=5, decoder = 4, add  k3n32s1+k3n32s1+k3n16s1 + k3n16s1 between k3n64s1 and k7n3s1  
    # try more layers
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        # ngf = 64
        assert(n_blocks >= 0)
        super(ResnetGenDecoder_upsample, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
    
        # residual block * n_blocks 
        # k3n256s1
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        # first upsampling
        # k3n128s1/2 + k3n128s1 
        model += [nn.Upsample(scale_factor=2, mode='bilinear'),
          nn.Conv2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1),
          nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        # second upsampling
        # k3n64s1/2 + k3n64s1
        model += [nn.Upsample(scale_factor=2, mode='bilinear'),
          nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1),
          nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * mult / 4)),
                      nn.ReLU(True)]

        # add addtional layers k3n32s1 + k3n32s1 + k3n16s1 + k3n16s1
        model += [  nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 8), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(ngf * mult / 8), int(ngf * mult / 8), kernel_size=3, stride=1, padding=1),
                    norm_layer(int(ngf * mult / 8)),
                    nn.ReLU(True)]
        model += [  nn.Conv2d(int(ngf * mult / 8), int(ngf * mult / 16), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(ngf * mult / 16), int(ngf * mult / 16), kernel_size=3, stride=1, padding=1),
                    norm_layer(int(ngf * mult / 16)),
                    nn.ReLU(True)]

        model += [nn.Conv2d(int(ngf * mult / 16), output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)
        
class ResnetGenDecoder_ILN_upblock(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder_ILN_upblock, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
  
 
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1),
                      ILN(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ConvTranspose2d(int(ngf * mult/2), int(ngf * mult / 4),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1),
                      ILN(int(ngf * mult / 4)),
                      nn.ReLU(True)]

        model += [  nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 8), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(ngf * mult / 8), int(ngf * mult / 8), kernel_size=3, stride=1, padding=1),
                    norm_layer(int(ngf * mult / 8)),
                    nn.ReLU(True)]
        model += [  nn.Conv2d(int(ngf * mult / 8), int(ngf * mult / 16), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(ngf * mult / 16), int(ngf * mult / 16), kernel_size=3, stride=1, padding=1),
                    norm_layer(int(ngf * mult / 16)),
                    nn.ReLU(True)]

        model += [nn.Conv2d(int(ngf * mult / 16), output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class ResnetGenDecoder_ILN_decoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
  
        assert(n_blocks >= 0)
        super(ResnetGenDecoder_ILN_decoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1),
                      ILN(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ConvTranspose2d(int(ngf * mult/2), int(ngf * mult / 4),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1),
                      ILN(int(ngf * mult / 4)),
                      nn.ReLU(True)]

        model += [  nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 8), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(ngf * mult / 8), int(ngf * mult / 8), kernel_size=3, stride=1, padding=1),
                    ILN(int(ngf * mult / 8)),
                    nn.ReLU(True)]
        model += [  nn.Conv2d(int(ngf * mult / 8), int(ngf * mult / 16), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(ngf * mult / 16), int(ngf * mult / 16), kernel_size=3, stride=1, padding=1),
                    ILN(int(ngf * mult / 16)),
                    nn.ReLU(True)]

        model += [nn.Conv2d(int(ngf * mult / 16), output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

# using adaIN normalization     
class ResnetGenDecoder_AdaIN_upblock(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
    # ngf = 64
        assert(n_blocks >= 0)
        super(ResnetGenDecoder_AdaIN_upblock, self).__init__()
        self.gpu_ids = gpu_ids

        resblock = []
        n_downsampling = 2
        mult = 2**n_downsampling
    
        for _ in range(n_blocks):
            resblock += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        first_upblock = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * mult / 2))]
        self.norm1 = AdaIN()
        second_upblock = [nn.ConvTranspose2d(int(ngf * mult/2), int(ngf * mult / 4),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * mult / 4))]
        self.norm2 = AdaIN()

        model = [nn.Conv2d(int(ngf * mult / 4), output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.resblock = nn.Sequential(*resblock)
        self.first_upblock = nn.Sequential(*first_upblock)
        self.second_upblock = nn.Sequential(*second_upblock)
        self.model = nn.Sequential(*model)
        self.activation = nn.ReLU(True)

    def forward(self, input, style1, style2):
        x_1 = self.resblock(input)
        
        x_2 = self.first_upblock(x_1)
        x_a_2 = self.activation(self.norm1(x_2, style1,True))

        x_3 = self.second_upblock(x_a_2)
        x_a_3 = self.activation(self.norm2(x_3, style2,True))
        
        return self.model(x_a_3)

# using adaIn norm in decoder
class ResnetGenDecoder_AdaIN_decoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
    # ngf = 64
        assert(n_blocks >= 0)
        super(ResnetGenDecoder_AdaIN_decoder, self).__init__()
        self.gpu_ids = gpu_ids

        resblock = []
        n_downsampling = 2
        mult = 2**n_downsampling
        self.n_blocks = n_blocks
    
        for i in range(n_blocks):
            setattr(self, 'ResnetBlock_' + str(i+1), ResnetAdaINBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type))
  
        first_upblock = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * mult / 2))]
        self.norm1 = AdaIN()
  
        second_upblock = [nn.ConvTranspose2d(int(ngf * mult/2), int(ngf * mult / 4),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * mult / 4))]
        self.norm2 = AdaIN()

        model = [nn.Conv2d(int(ngf * mult / 4), output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.resblock = nn.Sequential(*resblock)
        self.first_upblock = nn.Sequential(*first_upblock)
        self.second_upblock = nn.Sequential(*second_upblock)
        self.model = nn.Sequential(*model)
        self.activation = nn.ReLU(True)

    def forward(self, x, style1, style2, style3):
        # if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            # return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        gamma1, beta1 = torch.mean(style1, dim=[2, 3], keepdim=True), torch.var(style1, dim=[2, 3], keepdim=True)
        gamma2, beta2 = torch.mean(style2, dim=[2, 3], keepdim=True), torch.var(style2, dim=[2, 3], keepdim=True)
        gamma3, beta3 = torch.mean(style3, dim=[2, 3], keepdim=True), torch.var(style3, dim=[2, 3], keepdim=True)
        

        for i in range(self.n_blocks):
            x = getattr(self, 'ResnetBlock_' + str(i+1))(x, gamma1, beta1, True)
        x_1 = self.resblock(x)
        
        x_2 = self.first_upblock(x_1)
        x_a_2 = self.activation(self.norm1(x_2, gamma2, beta2, True))

        x_3 = self.second_upblock(x_a_2)
        x_a_3 = self.activation(self.norm2(x_3, gamma3, beta3, True))
        
        return self.model(x_a_3)

class ResnetGenDecoder(nn.Module):
    # encoder=5, decoder = 4, add  k3n32s1+k3n32s1+k3n16s1 + k3n16s1 between k3n64s1 and k7n3s1  
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
	# ngf = 64
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling
	
	# residual block * n_blocks 
	# k3n256s1
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

	# first upsampling
	# k3n128s1/2 + k3n128s1 
        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 2), int(ngf * mult / 2), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

	# second upsampling
	# k3n64s1/2 + k3n64s1
        model += [nn.ConvTranspose2d(int(ngf * mult/2), int(ngf * mult / 4),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1, bias=use_bias),
          nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 4), kernel_size=3, stride=1, padding=1),
                      norm_layer(int(ngf * mult / 4)),
                      nn.ReLU(True)]

	# add addtional layers k3n32s1 + k3n32s1 + k3n16s1 + k3n16s1
        model += [  nn.Conv2d(int(ngf * mult / 4), int(ngf * mult / 8), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(ngf * mult / 8), int(ngf * mult / 8), kernel_size=3, stride=1, padding=1),
                    norm_layer(int(ngf * mult / 8)),
                    nn.ReLU(True)]
        model += [  nn.Conv2d(int(ngf * mult / 8), int(ngf * mult / 16), kernel_size=3, stride=1, padding=1),
            nn.Conv2d(int(ngf * mult / 16), int(ngf * mult / 16), kernel_size=3, stride=1, padding=1),
                    norm_layer(int(ngf * mult / 16)),
                    nn.ReLU(True)]

        model += [nn.Conv2d(int(ngf * mult / 16), output_nc, kernel_size=7, stride=1, padding=3),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            print('use the dropout!!!!!!!')
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)

class ResnetAdaINBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetAdaINBlock, self).__init__()
        # self.conv_block = SequentialContext(n_domains, *conv_block)
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim + n_domains, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = AdaIN()
        self.relu1 = nn.ReLU()

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim + n_domains, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = AdaIN()

    def forward(self, input, gamma, beta, flag):
        out = self.pad1(input)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta, flag)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta, flag)

        return out + input

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.model_dis = self.model(input_nc, ndf, n_layers, norm_layer, use_sigmoid=False)

    def model(self, input_nc, ndf, n_layers, norm_layer, use_sigmoid=False):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences_common = []
        sequences_common += [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1), # k3n32s1
            nn.LeakyReLU(0.2, True)
        ]
       
        sequences_common += [
            nn.Conv2d(ndf, ndf * 2, 3, 2, 1), # k3n64s2 ken128s1
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1), 
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, 2,1), # k3n128 s2 k3n256s1
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, 1, 3, 1, 1)
        ]

        if use_sigmoid:
            sequences_common +=[nn.sigmoid()]

        return nn.Sequential(*sequences_common)
  

    def forward(self, input):
        # forward the input without any opreation
        output = self.model_dis(input)
        return output 

class Discriminator_class(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, blur_fn=None, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(Discriminator_class, self).__init__()
        self.gpu_ids = gpu_ids
        self.blur_fn = blur_fn
        self.gray_fn = lambda x: (.299*x[:,0,:,:] + .587*x[:,1,:,:] + .114*x[:,2,:,:]).unsqueeze_(1)
  # use gray or rgb images
        # self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_common, self.model_gan, self.model_class = self.model(input_nc, ndf, n_layers, norm_layer, use_sigmoid=False)

    def model(self, input_nc, ndf, n_layers, norm_layer, use_sigmoid=False):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences_common = []
        sequences_common += [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1), # k3n32s1
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
       
        sequences_common += [
      nn.Conv2d(ndf, ndf * 2, 3, 2, 1), # k3n64s2 ken128s1
      nn.LeakyReLU(0.2, True),
      nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1), 
      norm_layer(ndf * 4),
      nn.LeakyReLU(0.2, True),
      nn.Conv2d(ndf * 4, ndf * 4, 3, 2,1), # k3n128 s2 k3n256s1
      nn.LeakyReLU(0.2, True),
      nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
      norm_layer(ndf * 8),
      nn.LeakyReLU(0.2, True),
      nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1),
      norm_layer(ndf * 8),
      nn.LeakyReLU(0.2, True),
      nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1),
      norm_layer(ndf * 16),
      nn.LeakyReLU(0.2, True),
  ]

        sequences_gan =[]  # for gan loss 
        sequences_gan +=[
          nn.Conv2d(ndf * 16, 1, 3, 1, 1)
      ]

        sequences_class =[]  # for classification loss
        sequences_class += [
          nn.Conv2d(ndf * 16, 3, 18, 1, 1),
          nn.Sigmoid()
      ]

        if use_sigmoid:
            sequences +=[nn.sigmoid()]

        return nn.Sequential(*sequences_common), nn.Sequential(*sequences_gan), nn.Sequential(*sequences_class)
  

    def forward(self, input):
        temp = self.model_common(input)

        return self.model_gan(temp), self.model_class(temp)

class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        for net in self.networks:
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) for net in self.networks]

    def zero_grads(self, dom_a, dom_b, dom_c):
	# zero the grads of models
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()
        self.optimizers[dom_c].zero_grad()

    def zero_grads_one(self, dom):
        self.optimizers[dom].zero_grad()

    def step_grads(self, dom_a, dom_b, dom_c):
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()
        self.optimizers[dom_c].step()

    def step_grads_one(self, dom):
        self.optimizers[dom].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))

    def load_part(self, save_path, part):
        filename = save_path + ('%d.pth' % part)
        net.load_state_dict(torch.load(filename))

class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.n_domains = n_domains
        self.encoders = [encoder(*enc_args) for _ in range(1)]  # one common encoder
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]

        self.networks = self.encoders + self.decoders

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        # for the common encoder
        params = self.encoders[0].parameters()
        self.optimizers.append( opt(params, lr=lr, betas=betas))
        # for n_domains decoder
        for i in range(self.n_domains):
            params = self.decoders[i].parameters()
            self.optimizers.append( opt(params, lr=lr, betas=betas) )

    def forward(self, input, in_domain, out_domain):
        encoded = self.encode(input, 0)  # only use one encoder
        return self.decode(encoded,  out_domain)

    def encode(self, input, domain):
        output = self.encoders[domain].forward(input)
        return output

    def decode(self, input, domain):
        return self.decoders[domain].forward(input)

    def zero_grads(self, dom_a, dom_b, dom_c):
        self.optimizers[0].zero_grad()
        self.optimizers[dom_a+1].zero_grad()
        self.optimizers[dom_b+1].zero_grad()
        self.optimizers[dom_c+1].zero_grad()

    def step_grads(self, dom_a, dom_b, dom_c):
        self.optimizers[0].step()
        self.optimizers[dom_a+1].step()
        self.optimizers[dom_b+1].step()
        self.optimizers[dom_c+1].step()

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) +'\n'+ repr(d) +'\n'+ \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) +'\n'+ \
            'Number of parameters per Encoder: %d' % e_params +'\n'+ \
            'Number of parameters per Deocder: %d' % d_params

class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        self.n_domains = n_domains
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Discriminator: %d' % t_params

    def set_requires_grad(self, requires_grad=False):
        for i in range(self.n_domains):
            for param in self.networks[i].parameters():
                param.requires_grad = requires_grad


# class for VGG19 modle
class VGG19(nn.Module):
    def __init__(self, init_weights=None, feature_mode=False, batch_norm=False, num_classes=1000, gpu_id = []):
        super(VGG19, self).__init__()
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        self.init_weights = init_weights
        self.feature_mode = feature_mode
        self.batch_norm = batch_norm
        self.num_clases = num_classes
        self.features = self.make_layers(self.cfg, batch_norm)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        print(self.features)
        module_list = list(self.features.modules())
        print(module_list[1:9])

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:27]:                 # conv4_4
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x

    def forward_style(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:4]:   
                x = l(x)
            style1 = x
            for l in module_list[4:9]:
                x = l(x)
            style2 = x
            for l in module_list[9:18]:
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return style1, style2, x

    def forward_gram(self, x):
        if self.feature_mode:
            module_list = list(self.features.modules())
            for l in module_list[1:9]:   
                x = l(x)
        if not self.feature_mode:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)

        return x

class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x

class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;  continue
            predictions.append( output[:,-1,:,:] )
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]
        return predictions
