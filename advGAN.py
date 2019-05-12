import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os

models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model_1,
                 model_2,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model_1 = model_1
        self.model_2 = model_2
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        self.netDisc_1 = models.Discriminator(image_nc).to(device)
        self.netDisc_2 = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc_1.apply(weights_init)
        self.netDisc_2.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D_1 = torch.optim.Adam(self.netDisc_1.parameters(),
                                            lr=0.001)
        self.optimizer_D_2 = torch.optim.Adam(self.netDisc_2.parameters(),
                                            lr=0.001)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels):
        # optimize D1
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            # loss
            self.optimizer_D_1.zero_grad()
            pred_real_1 = self.netDisc_1(x)
            loss_D1_real = F.mse_loss(pred_real_1, torch.ones_like(pred_real_1, device=self.device))
            loss_D1_real.backward()

            pred_fake_1 = self.netDisc_1(adv_images.detach())
            loss_D1_fake = F.mse_loss(pred_fake_1, torch.zeros_like(pred_fake_1, device=self.device))
            loss_D1_fake.backward()
            loss_D1_GAN = loss_D1_fake + loss_D1_real

            self.optimizer_D_1.step()

        # optimize D2
        for i in range(1):
            perturbation = self.netG(x)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

            # loss
            self.optimizer_D_2.zero_grad()
            pred_real_2 = self.netDisc_2(x)
            loss_D2_real = F.mse_loss(pred_real_2, torch.ones_like(pred_real_2, device=self.device))
            loss_D2_real.backward()

            pred_fake_2 = self.netDisc_2(adv_images.detach())
            loss_D2_fake = F.mse_loss(pred_fake_2, torch.zeros_like(pred_fake_2, device=self.device))
            loss_D2_fake.backward()
            loss_D2_GAN = loss_D2_fake + loss_D2_real

            self.optimizer_D_2.step()

        loss_D_GAN = loss_D1_GAN + loss_D2_GAN

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = self.netDisc_1(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss 1
            logits_model = self.model_1(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv_1 = torch.max(real - other, zeros)
            loss_adv_1 = torch.sum(loss_adv_1)

            # cal adv loss 2
            logits_model = self.model_2(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]
            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1)
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
            zeros = torch.zeros_like(other)
            loss_adv_2 = torch.max(real - other, zeros)
            loss_adv_2 = torch.sum(loss_adv_2)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)
            loss_adv = (loss_adv_1 + loss_adv_2)*0.5


            adv_lambda = 10
            pert_lambda = 1
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D_1 = torch.optim.Adam(self.netDisc_1.parameters(),
                                                    lr=0.0001)
                self.optimizer_D_2 = torch.optim.Adam(self.netDisc_2.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D_1 = torch.optim.Adam(self.netDisc_1.parameters(),
                                                    lr=0.00001)
                self.optimizer_D_2 = torch.optim.Adam(self.netDisc_2.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%20==0:
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                torch.save(self.netG.state_dict(), netG_file_name)

