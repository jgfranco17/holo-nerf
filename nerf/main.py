import os 
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import numpy as np
import copy
import pickle
from PIL import Image
from tqdm import tqdm, trange
from torch.utils.data import DataLoader

from pointcloud import *
from utils import *

DEBUG = False
np.random.seed(0)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Running on {str(device).upper()}')
visualizer = PointCloudVisualizer()


class LFPatchDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True, file_dir = "./patch_data/Lego/"):
        if is_train:
            self.file_dir = os.path.join(file_dir, "train")
            self.file_list = []
            for f in sorted(os.listdir(self.file_dir)):
                self.file_list.append(os.path.join(self.file_dir, f))
            self.batch_num = len(self.file_list)
        else:
            self.batch_num = 1
            self.file_list = []
            self.file_list.append(os.path.join(file_dir, "patch_val.pkl"))

    def __len__(self):
        return self.batch_num

    def __getitem__(self, idx):
        filename_ = self.file_list[idx]
        with open(filename_, "rb") as f:
            ret_di= pickle.load(f)

        lab_t = torch.from_numpy(ret_di["y"]).float()
        lab_G_t = torch.from_numpy(ret_di["x"]).float()
    
        return lab_G_t, lab_t


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(coords, chunk=1024*32, use_viewdirs=False, **kwargs):

    """
    light  field coordinates(u,v,x,y) -> get xyz point's color and density
    """
    rays_o, rays_d = get_LF_rays(coords, **kwargs)

    if use_viewdirs:
        viewdirs = rays_d

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = kwargs["near"] * torch.ones_like(rays_d[...,:1]), kwargs["far"] * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render
    all_ret = batchify_rays(rays, chunk, **kwargs)

    for k in all_ret:
        if k != "ply":
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

    # if kwargs["make_plane"]:
    #     k_extract = ["rgb_map_cumsum", "raw"]
    #     ret_list = [all_ret[k] for k in k_extract]
    #     return ret_list

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'weights', 'depth_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}

    if kwargs["make_ply"]:
        ret_ply = all_ret["ply"]
        return ret_list + [ret_ply]
    return ret_list + [ret_dict] 


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips, 
                input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth, W=args.netwidth,
                    input_ch=input_ch, output_ch=output_ch, skips=skips).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start_epoch = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    testsavedir_for_compress = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start_epoch = ckpt['epoch']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'raw_noise_std' : args.raw_noise_std,
        "make_ply" : args.point_cloud,
        "lindisp" : args.lindisp
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start_epoch, grad_vars, optimizer

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, show_VS=False, fine = False, make_plane=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    #if show_VS and fine:
    #    make_VSmovie(dists, weights, rgb, raw)
    # if make_plane:
    #     rgb_each = weights[...,None]*rgb
    #     rgb_map_cumsum = torch.cumsum(rgb_each, dim = -2)#[N_rays,N_samples,3]
    #     return rgb_map_cumsum

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                make_ply=False,
                show_VolumeSampling=False,
                make_plane=False,
                near=-1,
                far = 1,
                xy_norm=1,
                z_factor=1.):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    #near : minimun disparity
    #far : maximum disparity

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # if make_plane:
    #     t_vals  = torch.linspace(0., 1., steps=128)
    #     z_vals = near * (1.-t_vals) + far * (t_vals)
    #     z_vals = z_vals.expand([N_rays, 128])
    #     pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    #     raw = network_query_fn(pts, viewdirs, network_fine)
    #     rgb_map_cumsum = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, make_plane=True)
    #     raw[...,:3] = torch.sigmoid(raw[...,:3])
    #     raw[...,3] = torch.sigmoid(raw[...,3])
    #     ret = {"rgb_map_cumsum":rgb_map_cumsum, "raw":raw}

    #     for k in ret:
    #         if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
    #             print(f"! [Numerical Error] {k} contains nan or inf.")
        
    #     return ret

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    # pts_copy = pts.cpu().numpy()
    # converted = pts_copy[0]
    # print(converted)
    # out_pcd = visualizer.create_pcd(converted)
    # visualizer.export_cloud(out_pcd, "outfiles/data.ply")

    # Normalization
    pts[...,2] = (pts[...,2] / xy_norm) * z_factor


#     raw = run_network(pts)

    if torch.max(pts) > 1.0:
        print("error", torch.max(pts))
        sys.exit(0)
    if torch.min(pts) < -1.0:
        print("error", torch.min(pts))
        sys.exit(0)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, show_VS = show_VolumeSampling, fine = False)

#     if N_importance > 0:

#         rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

#         z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
#         z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
#         z_samples = z_samples.detach()

#         z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
#         pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

#         run_fn = network_fn if network_fine is None else network_fine
# #         raw = run_network(pts, fn=run_fn)
#         raw = network_query_fn(pts, viewdirs, run_fn)

#         rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, show_VS = show_VolumeSampling, fine = True)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map': depth_map, 'weights' : weights}
    if retraw:
        ret['raw'] = raw
    # if N_importance > 0:
    #     ret['rgb0'] = rgb_map_0
    #     ret['disp0'] = disp_map_0
    #     ret['acc0'] = acc_map_0
    #     ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
    
    if make_ply:
        ply_raw = torch.cat([pts, raw], -1)
        ply_raw = torch.reshape(ply_raw, [-1,7])
        ret["ply"] = torch.sigmoid(ply_raw[:,:7])

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret



def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*16, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--num_epochs", type = int, default = 30,
                        help = "num epoch")

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    # parser.add_argument("--render_test", action='store_true', 
    #                     help='render the test set instead of render_poses path')

    # # training options
    # parser.add_argument("--precrop_iters", type=int, default=0,
    #                     help='number of steps to train on central crops')
    # parser.add_argument("--precrop_frac", type=float,
    #                     default=.5, help='fraction of img taken for central crops') 

    # dataset options
    # parser.add_argument("--testskip", type=int, default=8, 
    #                     help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    # parser.add_argument("--factor", type=int, default=8, 
    #                     help='downsample factor for LLFF images')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')

    parser.add_argument("--point_cloud", action = "store_true",
                        help = "if true, make point cloud")

    # logging/saving options
    parser.add_argument("--i_img",     type=int, default=2000, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=5, 
                        help='frequency of weight ckpt saving')

    # light field option
    parser.add_argument("--angRes", type = int, default = 17, help = "angular resolution")
    parser.add_argument("--img_H", type = int, default = 1024)
    parser.add_argument("--img_W", type = int, default = 1024)
    parser.add_argument("--trainset_dir", type = str, default = "LF_patches/lego_patch_9",
                        help = "trainset dir")
    
    parser.add_argument("--near", type = float, default = -5., help = "min z plane value")
    parser.add_argument("--far", type = float, default = 5., help = "max z plane value")

    parser.add_argument("--xy_norm", type = int, default=3, help = "xy plane normlization parameter, \
                            set max(abs(near, far))")
    parser.add_argument("--z_factor", type = float, default=1., help = "z plane factor")

    return parser


def train():
    print("Beginning data training...")
    parser = config_parser()
    args = parser.parse_args()

    basedir = args.basedir
    expname = args.expname
    arraydir = "./arraydata"
    depthdir = os.path.join(basedir, "maps")
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    testimgdir = os.path.join(basedir, expname, "val_imgs")
    os.makedirs(depthdir, exist_ok=True)
    os.makedirs(testimgdir, exist_ok = True)


    # Load data
    H = args.img_H
    W = args.img_W
    view_num = args.angRes
    trainset_dir = args.trainset_dir
    num_epochs = args.num_epochs

    near = args.near
    far = args.far

    # Create log dir and copy the config file
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        print(f'Reading config from: {f}')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start_epoch, grad_vars, optimizer = create_nerf(args)
    print("NeRF model initialized.")

    bds_dict = {
        'near' : near,
        'far' : far,
        "xy_norm" : args.xy_norm,
        "z_factor" : args.z_factor
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    trainset = LFPatchDataset(is_train=True, file_dir = trainset_dir)
    valset = LFPatchDataset(is_train=False, file_dir = trainset_dir)
    val_inp_t, val_lab_t = valset[0]

    bsize = 1
    train_loader = DataLoader(trainset, batch_size = bsize, drop_last=False, num_workers = 8, pin_memory=True)
    iters = len(train_loader)
    flag = 0
    print("Datasets loaded.")

    print("Start training!")
    mse_losses, psnrs, training_times = [], [], []

    for epoch in range(start_epoch, num_epochs):
        print(f'Running epoch {epoch}')
        e_psnr, e_loss, it = 0, 0, 0
        t = tqdm(train_loader)
        epoch_training_time = 0

        for batch_idx, (inp_G_t, lab_t) in enumerate(t):
            time0 = time.time()
            optimizer.zero_grad()
            inp_G_t, lab_t = inp_G_t.view(-1, inp_G_t.shape[-1]).to(device), lab_t.view(-1, 3).to(device)

            #(rgb) : (batch, 3)

            rgb, disp, acc, weights, depth, _ = render(inp_G_t,  chunk = args.chunk, use_viewdirs=args.use_viewdirs, **render_kwargs_train)

            mse_loss = torch.nn.functional.mse_loss(rgb, lab_t)
            loss = mse_loss
            loss.backward()
            optimizer.step()

            psnr = 10 * np.log10(1 / mse_loss.item())
            e_psnr += psnr
            e_loss += mse_loss.item()

            dt = time.time() - time0
            epoch_training_time = epoch_training_time + dt


            if it % args.i_img == 0:
                val_inp_t = val_inp_t.view(-1, val_inp_t.shape[-1]).to(device)
                val_lab_t = val_lab_t.view(-1, val_lab_t.shape[-1]).to(device)
                val_inp_t_c = copy.deepcopy(val_inp_t)

                b_size = val_inp_t_c.shape[0] // 4
                with torch.no_grad():
                    out = []
                    depths = []
                    for b in range(4):
                        rgb, disp, acc, weights, depth, _ = render(val_inp_t_c[b_size*b:b_size*(b+1)], chunk = args.chunk, use_viewdirs=args.use_viewdirs, **render_kwargs_test)
                        out.append(rgb)
                        depths.append(depth)
                        
                    out = torch.cat(out, dim=0)
                    out = torch.clamp(out, 0, 1)
                    out_np = out.view(H, W, 3).cpu().numpy() * 255
                    
                    depth = torch.cat(depths, dim=0)
                    max = torch.max(depth)
                    min = torch.min(depth)
                    # print(max, min)
                    depth = (depth - min) / (max - min)
                    depth = torch.clamp(depth, 0, 1)
                    depth_np = depth.view(H, W).cpu().numpy() * 255

                    # Generate output
                    out_np = out_np.astype(np.uint8)
                    output_npy_file = os.path.join(arraydir, f'array_e_{epoch}_it_{it}.npy')
                    np.save(output_npy_file, out_np) 
                    print(f'Exported {out_np.shape} array to {output_npy_file}')
                    depth_np = depth_np.astype(np.uint8)
                    depth_npy_file = os.path.join(depthdir, f'depth_e_{epoch}_it_{it}.npy')
                    np.save(depth_npy_file, depth_np) 
                    print(f'Exported {depth_np.shape} array to {depth_npy_file}')
                    print(depth_np)
                    out_im = Image.fromarray(np.uint8(out_np))
                    out_im.save(os.path.join(testimgdir, f'valout_e_{epoch}_it_{it}.png'))

            it += 1
            t.set_postfix(PSNR = psnr, EpochPSNR = e_psnr / it, EpochLoss = e_loss / it)

                    
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1   
        decay_steps = num_epochs * 1000

        new_lrate = args.lrate * (decay_rate ** (epoch / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################
        print('Epoch: %s Ave PSNR: %s Ave Loss: %s'%(epoch, e_psnr / it, e_loss / it))
        psnrs.append(e_psnr / it); mse_losses.append(e_loss / it); training_times.append(epoch_training_time)


        if epoch % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:02d}.tar'.format(epoch))
            torch.save({
                    'epoch': epoch,
                    'network_fn_state_dict': render_kwargs_train["network_fn"].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print("Saved checkpoints at", path)
    
    torch.save(render_kwargs_train["network_fn"].state_dict(), os.path.join(basedir, expname, "model.pth"))


    np.savetxt(os.path.join(basedir, expname, 'mse_stats.txt'), mse_losses, delimiter=',')
    np.savetxt(os.path.join(basedir, expname, 'psnr_stats.txt'), psnrs, delimiter=',')
    np.savetxt(os.path.join(basedir, expname, "training_time_stats.txt"), training_times, delimiter=",")
    all_time = 0
    for times in training_times:
        all_time = all_time + times

    np.savetxt(os.path.join(basedir, expname, "all_time.txt"), all_time, delimiter=",")


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()