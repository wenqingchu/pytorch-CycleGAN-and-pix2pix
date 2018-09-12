import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import deeplab_vgg

class DaStnModel(BaseModel):
    def name(self):
        return 'DaStnModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        # changing the default values to match the pix2pix paper
        # (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(pool_size=0)
        parser.set_defaults(no_lsgan=True)
        parser.set_defaults(norm='batch')
        # parser.set_defaults(dataset_mode='aligned')
        parser.set_defaults(dataset_mode='unaligned')
        #parser.set_defaults(which_model_netG='unet_256')
        opt, _ = parser.parse_known_args()
        #if opt.stn == 'unbounded_stn':
        #    parser.set_defaults(which_model_netG='unbounded_stn')
        #elif opt.stn == 'bounded_stn':
        #    parser.set_defaults(which_model_netG='bounded_stn')

        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=50.0, help='weight for L1 loss')

        return parser

    def initialize(self, opt):
        self.palette = np.array([128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                   220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
                   0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32])
        zero_pad = 256 * 3 - len(self.palette)
        for i in range(zero_pad):
            self.palette = np.append(self.palette,0)
        self.palette = self.palette.reshape((256,3))



        BaseModel.initialize(self, opt)
        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.isTrain = opt.isTrain
        self.which_model_netG = opt.which_model_netG
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_STN', 'Seg', 'D_real', 'D_fake', 'Target']
        #self.loss_names = ['G_GAN', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        #self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.netS = deeplab_vgg.DeeplabVGG(num_classes=self.input_nc)
        if opt.which_model_netG =='unbounded_stn' or opt.which_model_netG == 'bounded_stn':
            self.visual_names = ['real_A_color', 'real_A_color_grid', 'fake_B_color', 'real_B_color']
        else:
            self.visual_names = ['image_A_color', 'label_A_color', 'image_B_color', 'predicted_A_color', 'transformed_A_color', 'predicted_B_color']
            #self.visual_names = ['real_A_color', 'fake_B_color', 'real_B_color', 'real_C_color', 'fake_C_color']
            # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['S', 'G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['S']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.fineSize, opt.fineSize, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt.gan)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.gan)
            #self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
            #                              opt.which_model_netD,
            #                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            #self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, gan=opt.gan).to(self.device)
            self.criterionSeg = torch.nn.CrossEntropyLoss(ignore_index=19).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizers = []
            learning_rate_S = 2.5e-4
            momentum_S = 0.9
            weight_decay_S = 0.0005
            self.optimizer_S = torch.optim.SGD(self.netS.optim_parameters(),
                                               lr=learning_rate_S, momentum=momentum_S, weight_decay=weight_decay_S)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr*0.02, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr*0.02, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_S)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        one = torch.FloatTensor([1])
        mone = one * -1
        self.one = one.to(self.device)
        self.mone = mone.to(self.device)
        self.gan = opt.gan

    def test(self, opt, input):
        #print(input['C_paths'])
        real_A = input['C']
        input_label = torch.nn.functional.softmax(real_A, dim=1)
        self.real_A = input_label.to(self.device)
        if opt.which_model_netG == 'affine_stn':
            with torch.no_grad():
                self.recovered_A, self.predicted_theta = self.netG(self.real_A)
            print(self.predicted_theta)
            return 0
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % opt.which_model_netG)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        image_A = input['A' if AtoB else 'B']
        image_B = input['B' if AtoB else 'A']
        label_A = input['C']

        self.image_A = image_A.to(self.device)
        self.image_A_color = self.image_A

        self.image_B = image_B.to(self.device)
        self.image_B_color = self.image_B

        self.label_A = label_A.to(self.device)
        label_A_color = self.label_A.data[0].cpu().float().numpy()
        label_A_color = np.asarray(label_A_color, dtype=np.uint8)
        label_A_color_numpy = np.zeros((label_A_color.shape[0], label_A_color.shape[1], 3))
        for i in range(20):
            label_A_color_numpy[label_A_color == i] = self.palette[i]
        label_A_color = label_A_color_numpy.astype(np.uint8)
        label_A_color = label_A_color.transpose(2, 0, 1)
        label_A_color = label_A_color[np.newaxis, :]
        self.label_A_color = torch.from_numpy(label_A_color).to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.predicted_A = self.netS(self.image_A)
        self.predicted_B = self.netS(self.image_B)
        if self.which_model_netG == 'bounded_stn' or self.which_model_netG == 'unbounded_stn':
            self.fake_B, source_control_points = self.netG(self.real_A)
            real_A_color = self.real_A_color[0].cpu().float().numpy()
            #source_control_points = source_control_points[0].cpu().float().detach().numpy()
            source_control_points = source_control_points[0].cpu().float().detach()
            real_A_color = Image.fromarray(real_A_color.transpose(1,2,0).astype(np.uint8)).convert('RGB').resize((256, 256))
            canvas = Image.new(mode='RGB', size=(128 * 4, 128 * 4), color=(128, 128, 128))
            canvas.paste(real_A_color, (128, 128))
            #print(source_control_points.shape)
            source_points = (source_control_points + 1) / 2 * 256 + 128
            #print(source_points.shape)
            draw = ImageDraw.Draw(canvas)
            for x, y in source_points:
                draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(255, 0, 0))
            grid_size = 4
            #print(source_points.shape)
            source_points = source_points.view(grid_size, grid_size, 2)
            for j in range(grid_size):
                for k in range(grid_size):
                    x1, y1 = source_points[j, k]
                    if j > 0:  # connect to left
                        x2, y2 = source_points[j - 1, k]
                        draw.line((x1, y1, x2, y2), fill=(255, 0, 0))
                    if k > 0:  # connect to up
                        x2, y2 = source_points[j, k - 1]
                        draw.line((x1, y1, x2, y2), fill=(255, 0, 0))

            real_A_color = np.asarray(canvas.resize((256,256), Image.BICUBIC), np.uint8)
            real_A_color_grid = real_A_color.transpose(2, 0, 1)
            real_A_color_grid = real_A_color_grid[np.newaxis, :]
            self.real_A_color_grid = torch.from_numpy(real_A_color_grid).to(self.device)

        elif self.which_model_netG == 'affine_stn':
            self.transformed_A, theta_a = self.netG(self.predicted_A)
            print(theta_a)
        else:
            self.transformed_A= self.netG(self.real_A)
        # visualize the fake_B


        predicted_A_color = self.predicted_A.data[0].cpu().float().numpy()
        predicted_A_color = predicted_A_color.transpose(1, 2, 0)
        predicted_A_color = np.asarray(np.argmax(predicted_A_color, axis=2), dtype=np.uint8)
        predicted_A_color_numpy = np.zeros((predicted_A_color.shape[0], predicted_A_color.shape[1], 3))
        for i in range(20):
            predicted_A_color_numpy[predicted_A_color == i] = self.palette[i]
        predicted_A_color = predicted_A_color_numpy.astype(np.uint8)
        predicted_A_color = predicted_A_color.transpose(2, 0, 1)
        predicted_A_color = predicted_A_color[np.newaxis, :]
        self.predicted_A_color = torch.from_numpy(predicted_A_color).to(self.device)

        predicted_B_color = self.predicted_B.data[0].cpu().float().numpy()
        predicted_B_color = predicted_B_color.transpose(1, 2, 0)
        predicted_B_color = np.asarray(np.argmax(predicted_B_color, axis=2), dtype=np.uint8)
        predicted_B_color_numpy = np.zeros((predicted_B_color.shape[0], predicted_B_color.shape[1], 3))
        for i in range(20):
            predicted_B_color_numpy[predicted_B_color == i] = self.palette[i]
        predicted_B_color = predicted_B_color_numpy.astype(np.uint8)
        predicted_B_color = predicted_B_color.transpose(2, 0, 1)
        predicted_B_color = predicted_B_color[np.newaxis, :]
        self.predicted_B_color = torch.from_numpy(predicted_B_color).to(self.device)

        transformed_A_color = self.transformed_A.data[0].cpu().float().numpy()
        transformed_A_color = transformed_A_color.transpose(1, 2, 0)
        transformed_A_color = np.asarray(np.argmax(transformed_A_color, axis=2), dtype=np.uint8)
        transformed_A_color_numpy = np.zeros((transformed_A_color.shape[0], transformed_A_color.shape[1], 3))
        for i in range(20):
            transformed_A_color_numpy[transformed_A_color == i] = self.palette[i]
        transformed_A_color = transformed_A_color_numpy.astype(np.uint8)
        transformed_A_color = transformed_A_color.transpose(2, 0, 1)
        transformed_A_color = transformed_A_color[np.newaxis, :]
        self.transformed_A_color = torch.from_numpy(transformed_A_color).to(self.device)



    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        #fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        fake_B = self.fake_B_pool.query(self.transformed_A)
        #pred_fake = self.netD(fake_AB.detach())
        pred_fake = self.netD(fake_B.detach())
        #self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        #real_AB = torch.cat((self.real_A, self.real_B), 1)
        real_B = self.predicted_B
        #pred_real = self.netD(real_AB)
        pred_real = self.netD(real_B)
        if self.gan == 'vanilla' or self.gan == 'lsgan' or self.gan == 'sngan':
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # Combined loss
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
            self.loss_D.backward()
        elif self.gan == 'wgan' or self.gan == 'wgangp':
            tmp_pred_fake = pred_fake.mean()
            tmp_pred_real = pred_real.mean()
            #print(tmp_pred_fake.size())
            #print(tmp_pred_fake.data)
            #print(self.mone.size())
            tmp_pred_fake.backward(self.mone)
            tmp_pred_real.backward(self.one)
            self.loss_D_real = tmp_pred_real
            self.loss_D_fake = - tmp_pred_fake
            if self.gan == 'wgangp':
                # train with gradient penalty
                gradient_penalty = self.calc_gradient_penalty()
                gradient_penalty.backward()
        else:
            raise NotImplementedError('GANLoss name [%s] is not recognized' % self.gan)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        fake_B = self.transformed_A
        #pred_fake = self.netD(fake_AB)
        pred_fake = self.netD(fake_B)
        #self.loss_STN_L1 = self.criterionL1(self.predicted_theta, self.theta_i.float().to(self.device))
        if self.gan == 'vanilla' or self.gan == 'lsgan' or self.gan == 'sngan':
            self.loss_G_STN = self.criterionGAN(pred_fake, True)
            self.loss_G = self.loss_G_STN
            self.loss_G.backward()
        elif self.gan == 'wgan' or self.gan == 'wgangp':
            tmp_pred_fake = pred_fake.mean()
            tmp_pred_fake.backward(self.one)
            self.loss_G_GAN = tmp_pred_fake


    def backward_S(self):
        self.loss_Seg = self.criterionSeg(self.predicted_A, self.label_A)
        self.loss_Seg.backward()
        real_B = self.predicted_B
        pred_real = self.netD(real_B)
        if self.gan == 'vanilla' or self.gan == 'lsgan' or self.gan == 'sngan':
            self.loss_Target = self.criterionGAN(pred_real, False)
            self.loss_Target.backward()
        elif self.gan == 'wgan' or self.gan == 'wgangp':
            tmp_pred_real = pred_real.mean()
            tmp_pred_real.backward(self.mone)
            self.loss_Target = tmp_pred_real
            self.loss_Target.backward()


    def optimize_parameters(self):
        self.forward()
        # update D
        # clamp parameters to a cube
        if self.gan == 'wgan':
            clamp_lower = -0.01
            clamp_upper = 0.01
            for p in self.netD.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netS, False)
        self.set_requires_grad(self.netG, False)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netS, False)
        self.set_requires_grad(self.netG, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # update S
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG, False)
        self.set_requires_grad(self.netS, True)
        self.optimizer_S.zero_grad()
        self.backward_S()
        self.optimizer_S.step()


