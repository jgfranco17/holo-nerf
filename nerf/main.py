import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.quantization
from PIL import Image
from utils import *

# Configure runtime properties
DEBUG = False
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_LF_val(u, v, H, W, width=1024, height=1024):
    x = np.linspace(0, width-1, width)
    y = np.linspace(0, height-1, height)
    
    xv, yv = np.meshgrid(x, y)
    img_grid = torch.from_numpy(np.stack([yv, xv], axis=-1))
    uv_grid = torch.ones_like(img_grid)
    uv_grid[:, :, 0], uv_grid[:, :, 1] = v, u

    # Shift
    uv_grid = uv_grid - 8
    img_grid[...,0] = img_grid[...,0] - H //2
    img_grid[...,1] = img_grid[...,1] - W //2
    val_inp_t = torch.cat([uv_grid, img_grid], dim = -1).float()

    del img_grid, xv, yv
    return val_inp_t.view(-1, val_inp_t.shape[-1])


def batchify(fn, chunk):
    """
    Constructs a version of function that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """
    Prepares inputs and applies network 'fn'.
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
    """
    Render rays in smaller minibatches to avoid OOM.
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
    light  field coordinates (u,v,x,y) -> get xyz point's color and density
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

    # k_extract = ['rgb_map', 'disp_map', 'acc_map', "depth_map"]
    k_extract = ["rgb_map", "depth_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}

    if kwargs["make_ply"]:
        ret_ply = all_ret["ply"]
        return ret_list + [ret_ply]
    return ret_list + [ret_dict] 


def create_nerf(args):
    """
    Instantiate NeRF's MLP model.
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
    # if args.ft_path is not None and args.ft_path!='None':
    #     ckpts = [args.ft_path]
    # else:
    #     ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    # print('Found ckpts', ckpts)
    # testsavedir_for_compress = None
    # if len(ckpts) > 0 and not args.no_reload:
    #     ckpt_path = ckpts[-1]
    #     print('Reloading from', ckpt_path)
    #     ckpt = torch.load(ckpt_path)

    #     start_epoch = ckpt['epoch']
    #     optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    #     # Load model
    #     model.load_state_dict(ckpt['network_fn_state_dict'])
    #     if model_fine is not None:
    #         model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    model_path = torch.load(os.path.join(basedir, expname, "model.pth")) 
    model.load_state_dict(model_path)

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

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = float(0)

    return render_kwargs_train, render_kwargs_test, start_epoch, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
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

    rgb = torch.sigmoid(raw[...,:3])  # (N_rays, N_samples, 3)
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
    # disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # acc_map = torch.sum(weights, -1)

    # return rgb_map, disp_map, acc_map, weights, depth_map
    return rgb_map, depth_map
    # return rgb_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                white_bkgd=False,
                raw_noise_std=0.,
                pytest=False,
                make_ply=False,
                show_VolumeSampling=False,
                near=-1,
                far = 1,
                xy_norm=1,
                z_factor=1.):
    """
    Volumetric rendering.

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

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite 'u' with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand


    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    # Normalization of points in rays
    pts[...,2] = (pts[...,2] / xy_norm) * z_factor


    # raw = run_network(pts)

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, show_VS = show_VolumeSampling, fine = False)
    # rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest, show_VS = show_VolumeSampling, fine = False)

    ret = {"rgb_map" : rgb_map, "depth_map" : depth_map}
    if retraw:
        ret['raw'] = raw
    
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
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')

    parser.add_argument("--point_cloud", action = "store_true",
                        help = "if true, make point cloud")

    # logging/saving options
    parser.add_argument("--i_img",     type=int, default=1000, 
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


def main():
    # Configure args
    parser = config_parser()
    args = parser.parse_args()

    # Setup directories
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    testimgdir = os.path.join(basedir, expname, "val_imgs")
    os.makedirs(testimgdir, exist_ok = True)
    savedir = os.path.join(args.basedir, args.expname, "eval_output")
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(os.path.join(savedir, "all"), exist_ok=True)
    os.makedirs(os.path.join(savedir, "depth"), exist_ok=True)
    arraydir = "./arraydata/testdata"
    depthdir = "./arraydata/depthdata"
    colordir = "./arraydata/colordata"

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
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start_epoch, grad_vars, optimizer = create_nerf(args)

    bds_dict = {
        'near' : near,
        'far' : far,
        "xy_norm" : args.xy_norm,
        "z_factor" : args.z_factor
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    times = 0
    times2 = 0
    for v in tqdm(range(view_num)):
        for u in range(view_num):
            start = time.time()
            val_inp_t = get_LF_val(u=u, v=v, H=H, W=W, width=W, height =H).to(device)
            val_inp_t = val_inp_t.view(-1, val_inp_t.shape[-1]).to(device)

            b_size = val_inp_t.shape[0] // 4
            with torch.no_grad():
                out = []
                depths = []
                for b in range(4):
                    # rgb, disp, acc, depth, extras = render(val_inp_t[b_size*b:b_size*(b+1)]*1, chunk = args.chunk, use_viewdirs=args.use_viewdirs, **render_kwargs_test)
                    rgb, depth, _ = render(val_inp_t[b_size*b:b_size*(b+1)]*1, chunk = args.chunk, use_viewdirs=args.use_viewdirs, **render_kwargs_test)
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
                depth_np = depth.view(H, W).cpu().numpy()*255
                end1 = time.time()

                times += (end1 - start)
                # times2 += (end2 - start)

                out_np = out_np.astype(np.uint8)
                out_im = Image.fromarray(np.uint8(out_np))
                out_im.save(os.path.join(args.basedir, args.expname, "eval_output", "all", "re_{:02d}_{:02d}.png".format(v,u)))

                depth_np = depth_np.astype(np.uint8)
                output_npy_file = os.path.join(arraydir, f'array_e_{v}_{u}.npy')

                # Save color image data
                np.save(output_npy_file, out_np) 
                print(f'Exported {out_np.shape} array to {output_npy_file}')

                # Save depth map data
                depth_npy_file = os.path.join(depthdir, f'depth_e_{v}_{u}.npy')
                np.save(depth_npy_file, depth_np) 
                depth_npy_file = os.path.join(depthdir, f'depth_e_{v}_{u}.npy')
                np.save(depth_npy_file, depth_np) 
                print(f'DEPTH: Exported {depth_np.shape} array to {depth_npy_file}')
                print(depth_np)
                rgb_npy_file = os.path.join(colordir, f'color_e_{v}_{u}.npy')
                np.save(rgb_npy_file, out_np) 
                print(f'COLOR: Exported {out_np.shape} array to {rgb_npy_file}')
                print(out_np)

                # Generate depth image
                depth_im = Image.fromarray(np.uint8(depth_np))
                depth_im.save(os.path.join(args.basedir, args.expname, "eval_output", "depth", "re_{:02d}_{:02d}.png".format(v,u)))
    
    torch.save(render_kwargs_train["network_fn"].state_dict(), os.path.join(basedir, expname, "nerf.pth"))
    times = times / (view_num**2)
    # times2 = times2 / (17*17)
    print(f'Time elapsed: {times:.5f}')
    # print(times2)


if __name__=="__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()