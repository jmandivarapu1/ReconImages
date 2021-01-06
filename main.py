import torch
import torch.nn.functional as F
from Global import models

from Global.models import networks
from Global.models import mapping_model
import itertools
import torchvision

import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image as IMG
from torchvision import transforms as tmf
from Global.options.test_options import TestOptions


class CelebDataset(Dataset):
    def __init__(self, **kw):
        self.images_dir = kw.get('images_dir')
        self.images = os.listdir(self.images_dir)
        self.images = self.images[:kw.get('lim', len(self.images))]
        self.image_size = kw.get('image_size', 64)

    def __getitem__(self, index):
        file = self.images[index]
        img = self.transforms(IMG.open(self.images_dir + os.sep + file))
        return {'input': img}
    
    def __len__(self):
        return len(self.images)

    @property
    def transforms(self):
        return tmf.Compose(
            [tmf.Resize(self.image_size), tmf.CenterCrop(self.image_size),
             tmf.ToTensor(), tmf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

### lsgan: a=0, b=c=1
def lsgan_d(d_logit_real, d_logit_fake):
    return F.mse_loss(d_logit_real, torch.ones_like(d_logit_real)) + d_logit_fake.pow(2).mean()

def lsgan_g(d_logit_fake):
    return F.mse_loss(d_logit_fake, torch.ones_like(d_logit_fake))


def build_model(opt):
    """ stage 1.1  train 2 vae """
    # TODO stage 1.2 train mapping network
    print("build 2 vae and a transfer network")
    model = mapping_model.Pix2PixHDModel_Mapping()
    model.initialize(opt)

    ##### define networks
    print("build vae1 and vae2 ...")
    vae1 = networks.GlobalGenerator_DCDCv2(
        opt.input_nc,
        opt.output_nc,
        opt.ngf,
        opt.k_size,
        opt.n_downsample_global,
        networks.get_norm_layer(norm_type=opt.norm),
        opt=opt,
    )
    vae2 = networks.GlobalGenerator_DCDCv2(
        opt.input_nc,
        opt.output_nc,
        opt.ngf,
        opt.k_size,
        opt.n_downsample_global,
        networks.get_norm_layer(norm_type=opt.norm),
        opt=opt,
    )
    vae1.apply(networks.weights_init)
    vae2.apply(networks.weights_init)
    print("build vae1 and vae2 finish!")
    print("build D ...")
    xr_recon_d = networks.Z_xr_Discriminator(input_nc=3, ndf=opt.disc_ch, n_layers=opt.disc_layers).to(opt.device)
    z_xr_d = networks.Z_xr_Discriminator(input_nc=opt.feat_dim, ndf=opt.disc_ch, n_layers=opt.disc_layers).to(opt.device)
    y_recon_d =networks.Z_xr_Discriminator(input_nc=3, ndf=opt.disc_ch, n_layers=opt.disc_layers).to(opt.device)
    print("build D finish")
    """ Optim """
    optimizer_vae1 = torch.optim.Adam(vae1.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0, eps=1e-6)
    optimizer_d1 = torch.optim.Adam(itertools.chain(xr_recon_d.parameters(), z_xr_d.parameters()),lr=opt.lr, betas=(0.5, 0.999), weight_decay=0, eps=1e-6)
    optimizer_vae2 = torch.optim.Adam(vae2.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0, eps=1e-6)
    optimizer_d2 = torch.optim.Adam(y_recon_d.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0, eps=1e-6)
    return vae1, xr_recon_d, z_xr_d, vae2, y_recon_d, optimizer_vae1, optimizer_d1, optimizer_vae2, optimizer_d2

# build 2 vae network, 3 discriminators but NO transfer network for now
# and their optimizer


celebdataset = CelebDataset(images_dir='/data/akhanal1/img_align_celeba', lim=100)
dataloader = DataLoader(dataset=celebdataset, batch_size=4, pin_memory=True, num_workers=4)

batch = next(dataloader.__iter__())
print(batch['input'].shape)

sys.exit()
opt = TestOptions().parse(save=False)
parameter_set(opt)


# self.parser.add_argument("--batchSize", type=int, default=1, help="input batch size")
# self.parser.add_argument("--loadSize", type=int, default=1024, help="scale images to this size")
# self.parser.add_argument("--fineSize", type=int, default=512, help="then crop to this size")
# self.parser.add_argument("--label_nc", type=int, default=35, help="# of input label channels")
# self.parser.add_argument("--input_nc", type=int, default=3, help="# of input image channels")
# self.parser.add_argument("--output_nc", type=int, default=3, help="# of output image channels")
vae1 = networks.GlobalGenerator_DCDCv2(
        opt.input_nc,
        opt.output_nc,
        opt.ngf,
        opt.k_size,
        opt.n_downsample_global,
        networks.get_norm_layer(norm_type=opt.norm),
        opt=opt,
    )

vae1, xr_recon_d, z_xr_d, vae2, y_recon_d, optimizer_vae1, optimizer_d1, optimizer_vae2, optimizer_d2 = build_model(opt)
start_iter = 0
if opt.load_checkpoint_iter>0:
    checkpoint_path = checkpoint_root + f'/global_checkpoint_{opt.load_checkpoint_iter}.pth'
    if not Path(checkpoint_path).exists():
        print(f"ERROR! checkpoint_path {checkpoint_path} is None")
        exit(-1)
    state_dict = torch.load(checkpoint_path)
    start_iter = state_dict['iter']
    assert state_dict['batch_size'] == opt.batch_size, f"ERROR - batch size changed! load: {state_dict['batch_size']}, but now {opt.batch_size}"
    vae1.load_state_dict(state_dict['vae1'])
    xr_recon_d.load_state_dict(state_dict['xr_recon_d'])
    z_xr_d.load_state_dict(state_dict['z_xr_d'])
    vae2.load_state_dict(state_dict['vae2'])
    y_recon_d.load_state_dict(state_dict['y_recon_d'])
    optimizer_vae1.load_state_dict(state_dict['optimizer_vae1'])
    optimizer_d1.load_state_dict(state_dict['optimizer_d1'])
    optimizer_vae2.load_state_dict(state_dict['optimizer_vae2']) 
    optimizer_d2.load_state_dict(state_dict['optimizer_d2']) 
    print("checkpoint load successfully!")
# create dataloader
dataLoaderR, dataLoaderXY = get_dataloader(opt)
dataLoaderXY_iter = iter(dataLoaderXY)
dataLoaderR_iter = iter(dataLoaderR)
start = time.perf_counter()
print("train start!")
for ii in range(opt.total_iter - start_iter):
    current_iter = ii + start_iter
    try:
        x, y, path_y = dataLoaderXY_iter.next()
    except:
        dataLoaderXY_iter = iter(dataLoaderXY)
        x, y, path_y = dataLoaderXY_iter.next()
    try:
        r, path_r = dataLoaderR_iter.next()
    except:
        dataLoaderR_iter = iter(dataLoaderR)
        r, path_r = dataLoaderR_iter.next()
    ### following the practice in U-GAT-IT:
    ### train D and G iteratively, but not training D multiple times than training G
    r = r.to(opt.device)
    x = x.to(opt.device)
    y = y.to(opt.device)
    if opt.debug and current_iter%500==0:
        torchvision.utils.save_image(y, 'train_vae_y.png', normalize=True)
        torchvision.utils.save_image(x, 'train_vae_x.png', normalize=True)
        torchvision.utils.save_image(r, 'train_vae_r.png', normalize=True)

    ### vae1 train d
    # save gpu memory since no need calc grad for net G when train net D
    with torch.no_grad():
        z_x, mean_x, var_x, recon_x = vae1(x)
        z_r, mean_r, var_r, recon_r = vae1(r)
        batch_requires_grad(z_x, mean_x, var_x, recon_x,z_r, mean_r, var_r, recon_r)
    loss_1 = 0
    adv_loss_d_x = lsgan_d(xr_recon_d(x), xr_recon_d(recon_x))
    adv_loss_d_r = lsgan_d(xr_recon_d(r), xr_recon_d(recon_r))
    # z_x is real and z_r is fake here because let z_r close to z_x
    adv_loss_d_xr = lsgan_d(z_xr_d(z_x), z_xr_d(z_r))
    loss_1_d = adv_loss_d_x + adv_loss_d_r + adv_loss_d_xr
    loss_1_d.backward()
    optimizer_d1.step()
    optimizer_d1.zero_grad()
    ### vae1 train g
    # since we need update weights of G, the result should be re-calculate with grad
    z_x, mean_x, var_x, recon_x = vae1(x)
    z_r, mean_r, var_r, recon_r = vae1(r)
    adv_loss_g_x = lsgan_g(xr_recon_d(recon_x))
    adv_loss_g_r = lsgan_g(xr_recon_d(recon_r))
    # z_x is real and z_r is fake here because let z_r close to z_x
    adv_loss_g_xr = lsgan_g(z_xr_d(z_r))
    KLDloss_1_x = -0.5 * torch.sum(1 + var_x - mean_x.pow(2) - var_x.exp())  # KLD
    L1loss_1_x  = opt.weight_alpha * F.l1_loss(x, recon_x)
    KLDloss_1_r = -0.5 * torch.sum(1 + var_r - mean_r.pow(2) - var_r.exp())  # KLD
    L1loss_1_r  = opt.weight_alpha * F.l1_loss(r, recon_r)
    loss_1_g = adv_loss_g_x + KLDloss_1_x + L1loss_1_x + adv_loss_g_r + KLDloss_1_r + L1loss_1_r + adv_loss_g_xr
    loss_1_g.backward()
    optimizer_vae1.step()
    optimizer_vae1.zero_grad()

    ### vae2 train d
    # save gpu memory since no need calc grad for net G when train net D
    with torch.no_grad():
        z_y, mean_y, var_y, recon_y = vae2(y)
        batch_requires_grad(z_y, mean_y, var_y, recon_y)
    adv_loss_d_y = lsgan_d(y_recon_d(y), y_recon_d(recon_y))
    loss_2_d = adv_loss_d_y
    loss_2_d.backward()
    optimizer_d2.step()
    optimizer_d2.zero_grad()
    ### vae2 train g
    # since we need update weights of G, the result should be re-calculate with grad
    z_y, mean_y, var_y, recon_y = vae2(y)
    adv_loss_g_y = lsgan_g(y_recon_d(recon_y))
    KLDloss_1_y = -0.5 * torch.sum(1 + var_y - mean_y.pow(2) - var_y.exp())  # KLD
    L1loss_1_y  = opt.weight_alpha * F.l1_loss(y, recon_y)
    loss_2_g = adv_loss_g_y + KLDloss_1_y + L1loss_1_y
    loss_2_g.backward()
    optimizer_vae2.step()
    optimizer_vae2.zero_grad()
    # debug
    if opt.debug and current_iter%500==0:
        # [print(k, 'channel 0:\n', v[0][0]) for k,v in list(model.named_parameters()) if k in ["netG_A.encoder.13.conv_block.5.weight", "netG_A.decoder.4.conv_block.5.weight"]]
        torchvision.utils.save_image(recon_x, 'train_vae_recon_x.png', normalize=True)
        torchvision.utils.save_image(recon_r, 'train_vae_recon_r.png', normalize=True)
        torchvision.utils.save_image(recon_y, 'train_vae_recon_y.png', normalize=True)
    
    if current_iter%500==0:
        print(f"""STEP {current_iter:06d} {time.perf_counter() - start:.1f} s
        loss_1_d = adv_loss_d_x + adv_loss_d_r + adv_loss_d_xr
        {loss_1_d:.3f} = {adv_loss_d_x:.3f} + {adv_loss_d_r:.3f} + {adv_loss_d_xr:.3f}
        loss_1_g = adv_loss_g_x + KLDloss_1_x + L1loss_1_x + adv_loss_g_r + KLDloss_1_r + L1loss_1_r + adv_loss_g_xr
        {loss_1_g:.3f} = {adv_loss_g_x:.3f} + {KLDloss_1_x:.3f} + {L1loss_1_x:.3f} + {adv_loss_g_r:.3f} + {KLDloss_1_r:.3f} + {L1loss_1_r:.3f} + {adv_loss_g_xr:.3f}
        """)
    if (current_iter+1)%2000==0:
        # finish the current_iter-th step, e.g. finish iter0, save as 1, resume train from iter 1
        state = {
            'iter': current_iter,
            'batch_size': opt.batch_size,
            #
            'vae1': vae1.state_dict(),
            'xr_recon_d': xr_recon_d.state_dict(),
            'z_xr_d': z_xr_d.state_dict(),
            #
            'vae2': vae2.state_dict(),
            'y_recon_d': y_recon_d.state_dict(),
            #
            'optimizer_vae1': optimizer_vae1.state_dict(),
            'optimizer_d1': optimizer_d1.state_dict(),
            'optimizer_vae2': optimizer_vae2.state_dict(),
            'optimizer_d2': optimizer_d2.state_dict(),
            }
        torch.save(state, checkpoint_root + f'/global_checkpoint_{current_iter}.pth')
print("global", time.perf_counter() - start, ' s')