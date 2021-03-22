import numpy as np
import torch
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import copy
import torch.nn as nn

class CartoonGANModel(BaseModel):
    def name(self):
        return 'CartoonGANModel'

    def __init__(self, opt):
	# raise problems using super(),so use BaseModel.__init__(self.opt) instead
        # super(ComboGANModel, self).__init__(opt) 
        BaseModel.__init__(self, opt)
        self.n_domains = opt.n_domains
        self.d_domains = opt.d_domains
        self.batchSize = opt.batchSize
        self.DA, self.DB, self.DC = None, None, None  # classify the domains

        self.real = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.real_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)  # images in style 1
        self.real_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)  # images in style 2
        self.real_C = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)  # images in style 3
        # images without edges
        self.edge_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.edge_B = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.edge_C = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        # load/define networks
        self.netG = networks.define_G(opt.netG_framework, opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.netG_n_blocks, opt.netG_n_shared,
                                      self.n_domains, opt.norm, opt.use_dropout, self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(opt.netD_framework, opt.output_nc, opt.ndf, opt.netD_n_layers,
                                          self.d_domains, opt.norm, self.gpu_ids)
            self.classifier = networks.define_classifier(opt.classifier_framework, gpu_ids=self.gpu_ids) # for image classification
            self.vgg = networks.define_VGG(init_weights_=opt.vgg_pretrained_mode, feature_mode_=True, gpu_id_=self.gpu_ids) # using conv4_4 layer
        
        # load model weights
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain and not opt.init:
                self.load_network(self.netD, 'D', which_epoch)
                self.load_network(self.classifier, 'A', which_epoch)
            print("load weights of pretrained model successfully")

        # test the function of encoder part
        if opt.encoder_test:
            which_epoch = opt.which_epoch
            self.load_part_network(self.netG, 'G', which_epoch, 0)
            print("load weights of encoder successfully")

	# ======================training initialization==========================================
        if self.isTrain:
            self.fake_pools = [ImagePool(opt.pool_size) for _ in range(self.n_domains)]
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)  # use not opt.no_lsgan
            self.criterionContent = torch.nn.L1Loss()
            self.classGAN = networks.ClassLoss(tensor=self.Tensor)
            # initialize optimizers
            self.netG.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.netD.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))
            self.classifier.init_optimizers(torch.optim.Adam, opt.lr, (opt.beta1, 0.999))

            # initialize loss storage
            self.loss_D, self.loss_G_gan = [0]*self.n_domains, [0]*self.n_domains
            # discriminator loss in details
            self.loss_D_real = [0]*self.n_domains
            self.loss_D_fake = [0]*self.n_domains
            self.loss_D_edge = [0]*self.n_domains
            self.loss_D_class_real = [0]*self.n_domains
            self.loss_G_class = [0]*self.n_domains
            self.loss_D_class_edge_fake = [0]*self.n_domains
            # generator loss in details
            self.loss_content = [0]*self.n_domains
            self.loss_content_2 = [0] * self.n_domains
            self.loss_content_3 = [0] * self.n_domains
            # initialize loss multipliers
            self.lambda_con = opt.lambda_content
            self.lambda_cla = opt.lambda_classfication
	    

        print('---------- Networks initialized -------------')
        print(self.netG)
        if self.isTrain:
            print(self.netD)
            print(self.classifier)
        print('-----------------------------------------------')

    def set_input(self, input): # input is a dictionary recording images
        input_real = input['real']
        self.real.resize_(input_real.size()).copy_(input_real)
        self.image_paths = input['path_real']

        if self.isTrain:
            input_A = input['A']
            self.real_A.resize_(input_A.size()).copy_(input_A)
            self.DA = input['DA'][0]   # set the self.DA, self.DB
            input_B = input['B']
            self.real_B.resize_(input_B.size()).copy_(input_B)
            self.DB = input['DB'][0]
            input_C =input['C']
            self.real_C.resize_(input_C.size()).copy_(input_C)
            self.DC = input['DC'][0]
            # for images without edges
            edge_A = input['edge_A']
            self.edge_A.resize_(edge_A.size()).copy_(edge_A)
            edge_B = input['edge_B']
            self.edge_B.resize_(edge_B.size()).copy_(edge_B)
            edge_C = input['edge_C']
            self.edge_C.resize_(edge_C.size()).copy_(edge_C)

    def test(self):
        with torch.no_grad():
            self.visuals = [self.real]
            self.labels = ['real']
            encoded = self.netG.encode(self.real, 0)
            print(self.real.size())
            for d in range(self.n_domains):
                if d == self.DA and not self.opt.autoencode:
                    continue
                fake = self.netG.decode(encoded, d)
                self.visuals.append( fake )
                self.labels.append( 'fake_%d' % (d+1) )
                if self.opt.reconstruct:
                    rec = self.netG.forward(fake, d, self.DA)
                    self.visuals.append( rec )
                    self.labels.append( 'rec_%d' % d )

    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, real, fake, edge, domain):
        # D(real)
        pred_real = self.netD.forward(real, domain)
        loss_D_real = self.criterionGAN(pred_real, True)
        # D(fake)
        pred_fake = self.netD.forward(fake.detach(), domain)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # D(edge)
        pred_edge_fake = self.netD.forward(edge, domain)
        loss_D_edge_fake = self.criterionGAN(pred_edge_fake, False)

        # class loss
        real_logits = self.classifier(real)
        class_loss = self.classGAN(real_logits, domain)
        fake_edge_logits = self.classifier(edge) # edge -> class 3
        fake_edge_class_loss = self.classGAN(fake_edge_logits, 3)

        loss_D = (loss_D_real + loss_D_fake + loss_D_edge_fake ) / 3 + (class_loss + fake_edge_class_loss) / 2 * self.lambda_cla 
        # backward
        loss_D.backward()
        return loss_D_real, loss_D_fake, loss_D_edge_fake, class_loss, fake_edge_class_loss


    def backward_D(self):
        fake_A = self.fake_pools[self.DA].query(self.fake_A)  # choose the corresponding fake_pools
        fake_B = self.fake_pools[self.DB].query(self.fake_B)
        fake_C = self.fake_pools[self.DC].query(self.fake_C)

        self.loss_D_real[self.DA], self.loss_D_fake[self.DA],self.loss_D_edge[self.DA], self.loss_D_class_real[self.DA], self.loss_D_class_edge_fake[self.DA] = self.backward_D_basic(self.real_A, fake_A, self.edge_A, self.DA)
        self.loss_D_real[self.DB], self.loss_D_fake[self.DB],self.loss_D_edge[self.DB], self.loss_D_class_real[self.DB], self.loss_D_class_edge_fake[self.DB] = self.backward_D_basic(self.real_B, fake_B, self.edge_B, self.DB)
        self.loss_D_real[self.DC], self.loss_D_fake[self.DC],self.loss_D_edge[self.DC], self.loss_D_class_real[self.DC], self.loss_D_class_edge_fake[self.DC] = self.backward_D_basic(self.real_C, fake_C, self.edge_C, self.DC)
        
	
    def forward(self):
        encoded_A = self.netG.encode(self.real, 0)

        self.fake_A = self.netG.decode(encoded_A, self.DA).clone()
        self.fake_B = self.netG.decode(encoded_A, self.DB).clone()
        self.fake_C = self.netG.decode(encoded_A, self.DC).clone()


    def backward_G(self, init):
        if not init:
            # GAN loss
            pred_fake = self.netD.forward(self.fake_A, self.DA)   # D_A(G_A(input))
            self.loss_G_gan[self.DA] = self.criterionGAN(pred_fake, True)   
            pred_fake = self.netD.forward(self.fake_B, self.DB)   # D_B(G_B(input))
            self.loss_G_gan[self.DB] = self.criterionGAN(pred_fake, True)
            pred_fake = self.netD.forward(self.fake_C, self.DC)   # D_C(G_C(input))
            self.loss_G_gan[self.DC] = self.criterionGAN(pred_fake, True)

            # cla loss
            fake_logit_A = self.classifier(self.fake_A)
            self.loss_G_class[self.DA] = self.classGAN(fake_logit_A, self.DA)
            fake_logit_B = self.classifier(self.fake_B)
            self.loss_G_class[self.DB] = self.classGAN(fake_logit_B, self.DB)
            fake_logit_C = self.classifier(self.fake_C)
            self.loss_G_class[self.DC] = self.classGAN(fake_logit_C, self.DC)

        # content loss
        # 256 * 256 content loss using vgg
        x_a_feature = self.vgg(self.real)
        g_a_feature = self.vgg(self.fake_A)
        self.loss_content[self.DA] = self.criterionContent(g_a_feature, x_a_feature.detach())
        x_b_feature = self.vgg(self.real)
        g_b_feature = self.vgg(self.fake_B)
        self.loss_content[self.DB] = self.criterionContent(g_b_feature, x_b_feature.detach())
        x_c_feature = self.vgg(self.real)
        g_c_feature = self.vgg(self.fake_C)
        self.loss_content[self.DC] = self.criterionContent(g_c_feature, x_c_feature.detach())
        # 128*128 content loss
        x_a_feature_2 = nn.functional.interpolate(self.real, scale_factor=0.5, mode='bilinear') 
        x_a_feature_2 = nn.functional.interpolate(x_a_feature_2, scale_factor=2, mode='bilinear') 
        fake_A_2 = nn.functional.interpolate(self.fake_A, scale_factor=0.5, mode='bilinear') 
        fake_A_2 = nn.functional.interpolate(fake_A_2, scale_factor=2, mode='bilinear') 
        self.loss_content_2[self.DA] = self.criterionContent(fake_A_2, x_a_feature_2.detach())
        fake_B_2 = nn.functional.interpolate(self.fake_B, scale_factor=0.5, mode='bilinear') 
        fake_B_2 = nn.functional.interpolate(fake_B_2, scale_factor=2, mode='bilinear') 
        self.loss_content_2[self.DB] = self.criterionContent(fake_B_2, x_a_feature_2.detach())
        fake_C_2 = nn.functional.interpolate(self.fake_C, scale_factor=0.5, mode='bilinear') 
        fake_C_2 = nn.functional.interpolate(fake_C_2, scale_factor=2, mode='bilinear') 
        self.loss_content_2[self.DC] = self.criterionContent(fake_C_2, x_a_feature_2.detach())
        # 64*64 content loss 
        x_a_feature_3 = nn.functional.interpolate(self.real, scale_factor=0.25, mode='bilinear') 
        x_a_feature_3 = nn.functional.interpolate(x_a_feature_3, scale_factor=4, mode='bilinear') 
        fake_A_3 = nn.functional.interpolate(self.fake_A, scale_factor=0.25, mode='bilinear') 
        fake_A_3 = nn.functional.interpolate(fake_A_3, scale_factor=4, mode='bilinear') 
        self.loss_content_3[self.DA] = self.criterionContent(fake_A_3, x_a_feature_3.detach())
        fake_B_3 = nn.functional.interpolate(self.fake_B, scale_factor=0.25, mode='bilinear') 
        fake_B_3 = nn.functional.interpolate(fake_B_3, scale_factor=4, mode='bilinear') 
        self.loss_content_3[self.DB] = self.criterionContent(fake_B_3, x_a_feature_3.detach())
        fake_C_3 = nn.functional.interpolate(self.fake_C, scale_factor=0.25, mode='bilinear') 
        fake_C_3 = nn.functional.interpolate(fake_C_3, scale_factor=4, mode='bilinear') 
        self.loss_content_3[self.DC] = self.criterionContent(fake_C_3, x_a_feature_3.detach())
        
        if not init:
            loss_G = self.loss_G_gan[self.DA] + self.loss_G_gan[self.DB] + self.loss_G_gan[self.DC] + \
            (self.loss_G_class[self.DA] + self.loss_G_class[self.DB] + self.loss_G_class[self.DC]) / 3 * self.lambda_cla   + \
            (self.loss_content[self.DA] + self.loss_content[self.DB] + self.loss_content[self.DC] + \
            self.loss_content_2[self.DA] + self.loss_content_2[self.DB] + self.loss_content_2[self.DC] + \
            self.loss_content_3[self.DA] + self.loss_content_3[self.DB] + self.loss_content_3[self.DC]  ) / 3 * self.lambda_con
        else:
        # init phase
            loss_G = (self.loss_content[self.DA] + self.loss_content[self.DB] + self.loss_content[self.DC] + \
            self.loss_content_2[self.DA] + self.loss_content_2[self.DB] + self.loss_content_2[self.DC] + \
            self.loss_content_3[self.DA] + self.loss_content_3[self.DB] + self.loss_content_3[self.DC]  ) / 3 * self.lambda_con
        loss_G.backward()



    def optimize_parameters(self, init): # for initialization phase using optimize of generator only
        self.forward()
        # train G

        self.netD.set_requires_grad(False)
        self.classifier.set_requires_grad(False)
        self.netG.zero_grads(self.DA, self.DB, self.DC)
        self.backward_G(init = init)
        self.netG.step_grads(self.DA, self.DB, self.DC)

        if not init:
            # train D and auxiliary classifier
            self.netD.set_requires_grad(True)
            self.classifier.set_requires_grad(True)
            self.netD.zero_grads(self.DA, self.DB, self.DC)
            self.classifier.zero_grads()
            self.backward_D()
            self.netD.step_grads(self.DA, self.DB, self.DC)
            self.classifier.step_grads()
            
    def get_current_errors(self):
        # output the loss 
        extract = lambda l: [(i if type(i) is int or type(i) is float else i.item()) for i in l]
        D_real, D_fake, D_edge, D_class_real, D_class_edge,G_gan,G_class, G_content = extract(self.loss_D_real), extract(self.loss_D_fake), extract(self.loss_D_edge), extract(self.loss_D_class_real), extract(self.loss_D_class_edge_fake), extract(self.loss_G_gan), extract(self.loss_G_class),extract(self.loss_content)
        return OrderedDict([('D_real',D_real), ('D_fake',D_fake), ('D_edge', D_edge), ('class_real',D_class_real),('class_edge', D_class_edge), ('G_gan', G_gan), ('G_class', G_class),('Con', G_content)])

    def get_current_visuals(self, testing=False):
        if not testing:
            # self.visuals = [self.fake_A, self.fake_B, self.fake_C]
            # self.labels = ['fake' + str(self.DA), 'fake' + str(self.DB), 'fake'+str(self.DC)]
            self.visuals = [self.real, self.fake_A, self.fake_B, self.fake_C]
            self.labels = ['real', 'fake' + str(self.DA), 'fake' + str(self.DB), 'fake'+str(self.DC)]    
        images = [util.tensor2im(v.data) for v in self.visuals]
        return OrderedDict(zip(self.labels, images))

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.classifier, 'A', label, self.gpu_ids)

    def update_hyperparams(self, curr_iter):
        # for inititialization phase
        if curr_iter < self.opt.init_niter:
            new_lr = self.opt.init_lr
        elif curr_iter > self.opt.niter:  # update the learning rate in training phase
            decay_frac = (curr_iter - self.opt.niter) / self.opt.niter_decay
            new_lr = self.opt.lr * (1 - decay_frac)
            print('updated learning rate: %f' % new_lr)
        else:
            new_lr =self.opt.lr
        print(new_lr)
        self.netG.update_lr(new_lr)
        self.netD.update_lr(new_lr)
        self.classifier.update_lr(new_lr)


