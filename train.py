import torch
import os
import sys
import config
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
from dataset import HorseZebraDataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


def train(disc_H,disc_Z,gen_Z,gen_H,loader,opt_disc,opt_gen,l1,mse,d_scaler,g_scaler):
    loop = tqdm(loader,leave=True)

    for idx,(zebra,horse) in enumerate(loop):
        zebra = zebra.to(config.DEVICE)
        horse = horse.to(config.DEVICE)

        #Train discriminator
        with torch.cuda.amp.autocast():
            fake_horse = gen_H(zebra)
            D_H_real = disc_H(horse)
            D_H_fake = disc_H(fake_horse.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = D_H_real_loss + D_H_fake_loss

            fake_zebra = gen_Z(horse)
            D_Z_real = disc_Z(zebra)
            D_Z_fake = disc_Z(fake_zebra.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = D_Z_real_loss + D_Z_fake_loss

            D_loss = (D_Z_loss + D_H_loss)/2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        #Train generator
        with torch.cuda.amp.autocast():
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            G_H_loss = mse(D_H_fake, torch.ones_like(D_H_fake))
            G_Z_loss = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(cycle_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            G_loss = (
                G_H_loss+
                G_Z_loss+
                cycle_zebra_loss * config.LAMBDA_CYCLE+
                cycle_horse_loss * config.LAMBDA_CYCLE
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse*0.5+0.5, f'saved_images/horse_{idx}.png')
            save_image(fake_zebra*0.5+0.5, f'saved_images/zebra_{idx}.png')
         
def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channels=3,num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        beta=(0.5,0.999)
    )
    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN_H,gen_H,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_GEN_Z,gen_Z,opt_gen,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_H,disc_H,opt_disc,config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_CRITIC_Z,disc_Z,opt_disc,config.LEARNING_RATE)

    train_dataset = HorseZebraDataset(horse_root=config.TRAIN_DIR + '/horse',zebra_root=config.TRAIN_DIR + '/zebra',transform=config.transforms)
    test_dataset = HorseZebraDataset(horse_root='D:\\GAN_image_translation\\archive (3)\\test\\horse',zebra_root='D:\\GAN_image_translation\\archive (3)\\test\\zebra',transform=config.transforms)

    test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,pin_memory=True)
    loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,pin_memory=True)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config. NUM_EPOCHS):
        train(disc_H,disc_Z,gen_Z,gen_H,loader,opt_disc,opt_gen,L1,mse,d_scaler,g_scaler)

    if config.SAVE_MODEL:
        save_checkpoint(gen_H,opt_gen,filename=config.CHECKPOINT_GEN_H)
        save_checkpoint(gen_Z,opt_gen,filename=config.CHECKPOINT_GEN_Z)
        save_checkpoint(disc_H,opt_disc,filename=config.CHECKPOINT_CRITIC_H)
        save_checkpoint(disc_Z,opt_disc,filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == '__main__':
    main()