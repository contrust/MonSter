import os
import hydra
import torch
from tqdm import tqdm
import torch.optim as optim
# from util import InputPadder
from core.utils.utils import InputPadder
from core.monster import Monster 
from omegaconf import OmegaConf
import torch.nn.functional as F
from accelerate import Accelerator
import core.stereo_datasets as datasets
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import DataLoaderConfiguration
from accelerate.utils import DistributedDataParallelKwargs


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def gray_2_colormap_np(img, cmap = 'rainbow', max = None):
    img = img.cpu().detach().numpy().squeeze()
    assert img.ndim == 2
    img[img<0] = 0
    mask_invalid = img < 1e-10
    if max == None:
        img = img / (img.max() + 1e-8)
    else:
        img = img/(max + 1e-8)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
    cmap_m = matplotlib.cm.get_cmap(cmap)
    map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
    colormap = (map.to_rgba(img)[:,:,:3]*255).astype(np.uint8)
    colormap[mask_invalid] = 0

    return colormap

def sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, loss_gamma=0.9, max_disp=192):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(disp_preds)
    assert n_predictions >= 1
    disp_loss = 0.0
    mag = torch.sum(disp_gt**2, dim=1).sqrt()
    valid = ((valid >= 0.5) & (mag < max_disp)).unsqueeze(1)
    assert valid.shape == disp_gt.shape, [valid.shape, disp_gt.shape]
    assert torch.isfinite(disp_gt[valid.bool()]).all()

    # quantile = torch.quantile((disp_init_pred - disp_gt).abs(), 0.9)
    init_valid = valid.bool() & torch.isfinite(disp_init_pred) & torch.isfinite(disp_gt) #  & ((disp_init_pred - disp_gt).abs() < quantile)
    if init_valid.sum() == 0:
        print(f"Warning: no valid pixels for initial prediction loss calculation")
    else:
        disp_loss += 1.0 * F.smooth_l1_loss(disp_init_pred[init_valid], disp_gt[init_valid], reduction='mean')
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (disp_preds[i] - disp_gt).abs()
        # quantile = torch.quantile(i_loss, 0.9)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, disp_gt.shape, disp_preds[i].shape]
        mask = valid.bool() & torch.isfinite(i_loss)
        if mask.sum() == 0:
            print(f"Warning: no valid pixels for prediction loss calculation at iteration {i}")
            continue
        disp_loss += i_weight * i_loss[mask].mean()

    epe = torch.sum((disp_preds[-1] - disp_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    if valid.bool().sum() == 0:
        epe = torch.Tensor([0.0]).cuda()

    metrics = {
        'train/epe': epe.mean(),
        'train/1px': (epe < 1).float().mean(),
        'train/3px': (epe < 3).float().mean(),
        'train/5px': (epe < 5).float().mean(),
    }
    return disp_loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    DPT_params = list(map(id, model.feat_decoder.parameters())) 
    rest_params = filter(lambda x:id(x) not in DPT_params and x.requires_grad, model.parameters())

    params_dict = [{'params': model.feat_decoder.parameters(), 'lr': args.lr/2.0}, 
                   {'params': rest_params, 'lr': args.lr}, ]
    optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_step, eta_min=args.lr/100.0)

    return optimizer, scheduler

@hydra.main(version_base=None, config_path='config', config_name='train_us3d')
def main(cfg):
    set_seed(cfg.seed)
    logger = get_logger(__name__)
    Path(cfg.save_path).mkdir(exist_ok=True, parents=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision=None, dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True), log_with='tensorboard', project_dir=cfg.project_dir, kwargs_handlers=[kwargs], step_scheduler_with_optimizer=False)
       # Flatten config arrays into individual scalar hyperparameters for TensorBoard
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    hparams_config = {}
    
    for key, value in config_dict.items():
        if isinstance(value, (int, float, str, bool)):
            # Keep scalars as-is
            hparams_config[key] = value
        elif torch.is_tensor(value):
            # Keep tensors as-is (though they may not display well in TensorBoard)
            hparams_config[key] = value
        elif isinstance(value, (list, tuple)):
            # Flatten arrays into individual scalar parameters
            for i, item in enumerate(value):
                if isinstance(item, (int, float, str, bool)):
                    hparams_config[f"{key}{i}"] = item
                else:
                    # Convert complex nested items to strings
                    hparams_config[f"{key}{i}"] = str(item)
        else:
            # Convert other complex types (dicts, etc.) to strings
            hparams_config[key] = str(value)
    
    accelerator.init_trackers(project_name=cfg.project_name, config=hparams_config, init_kwargs={'tensorboard': cfg.tensorboard})

    dataset = datasets.fetch_dataloader(cfg)
    train_dataset, val_dataset, _ = torch.utils.data.random_split(dataset, [cfg.train_split_ratio, cfg.val_split_ratio, 1 - cfg.train_split_ratio - cfg.val_split_ratio])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=True, num_workers=1, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size//cfg.num_gpu,
        pin_memory=True, shuffle=False, num_workers=1, drop_last=False)

    aug_params = {}

    model = Monster(cfg)
    if not cfg.restore_ckpt.endswith("None"):
        assert cfg.restore_ckpt.endswith(".pth")
        print(f"Loading checkpoint from {cfg.restore_ckpt}")
        assert os.path.exists(cfg.restore_ckpt)
        checkpoint = torch.load(cfg.restore_ckpt, map_location='cpu')
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt[key.replace('module.', '')] = checkpoint[key]

        model.load_state_dict(ckpt, strict=True)
        print(f"Loaded checkpoint from {cfg.restore_ckpt} successfully")
        del ckpt, checkpoint
    optimizer, lr_scheduler = fetch_optimizer(cfg, model)
    train_loader, model, optimizer, lr_scheduler, val_loader = accelerator.prepare(train_loader, model, optimizer, lr_scheduler, val_loader)
    model.to(accelerator.device)

    total_step = 0
    should_keep_training = True
    while should_keep_training:
        active_train_loader = train_loader

        if (total_step % cfg.val_frequency == 0):
            model.eval()
            elem_num, total_epe, total_out = 0, 0, 0
            for data in tqdm(val_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
                _, left, right, disp_gt, valid = [x for x in data]
                padder = InputPadder(left.shape, divis_by=32)
                left, right = padder.pad(left, right)
                with torch.no_grad():
                    with accelerator.autocast():
                        disp_pred = model(left, right, iters=cfg.valid_iters, test_mode=True)
                disp_pred = padder.unpad(disp_pred)
                assert disp_pred.shape == disp_gt.shape, (disp_pred.shape, disp_gt.shape)
                epe = torch.abs(disp_pred - disp_gt)
                out = (epe > 1.0).float()
                epe = torch.squeeze(epe, dim=1)
                out = torch.squeeze(out, dim=1)
                disp_gt = torch.squeeze(disp_gt, dim=1)
                epe, out = accelerator.gather_for_metrics((epe[(valid >= 0.5) & (disp_gt.abs() < 192)].mean(), out[(valid >= 0.5) & (disp_gt.abs() < 192)].mean()))
                elem_num += epe.shape[0]
                for i in range(epe.shape[0]):
                    total_epe += epe[i]
                    total_out += out[i]
                accelerator.log({'val/epe': total_epe / elem_num, 'val/d1': 100 * total_out / elem_num}, total_step)


        model.train()
        model.module.freeze_bn()
        for data in tqdm(active_train_loader, dynamic_ncols=True, disable=not accelerator.is_main_process):
            _, left, right, disp_gt, valid = [x for x in data]
            with accelerator.autocast():
                disp_init_pred, disp_preds, depth_mono = model(left, right, iters=cfg.train_iters)
            loss, metrics = sequence_loss(disp_preds, disp_init_pred, disp_gt, valid, max_disp=cfg.max_disp)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            total_step += 1
            loss = accelerator.reduce(loss.detach(), reduction='mean')
            metrics = accelerator.reduce(metrics, reduction='mean')
            accelerator.log({'train/loss': loss, 'train/learning_rate': optimizer.param_groups[0]['lr']}, total_step)
            accelerator.log(metrics, total_step)

            ####visualize the depth_mono and disp_preds
            if total_step % cfg.image_visual_frequency == 0 and accelerator.is_main_process:
                image1_np = left[0].squeeze().cpu().numpy()
                image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
                image1_np = image1_np.astype(np.uint8)
                image1_np = np.transpose(image1_np, (1, 2, 0))

                image2_np = right[0].squeeze().cpu().numpy()
                image2_np = (image2_np - image2_np.min()) / (image2_np.max() - image2_np.min()) * 255.0
                image2_np = image2_np.astype(np.uint8)
                image2_np = np.transpose(image2_np, (1, 2, 0))


                depth_mono_np = gray_2_colormap_np(depth_mono[0].squeeze())
                disp_preds_np = gray_2_colormap_np(disp_preds[-1][0].squeeze())
                disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
                
                tracker = accelerator.get_tracker('tensorboard')
                if tracker is not None:
                    writer = tracker.writer
                    if writer is not None:
                        # Convert numpy arrays to tensors and transpose for tensorboard (HWC -> CHW)
                        writer.add_image('disp_pred', np.transpose(disp_preds_np, (2, 0, 1)), total_step, dataformats='CHW')
                        writer.add_image('disp_gt', np.transpose(disp_gt_np, (2, 0, 1)), total_step, dataformats='CHW')
                        writer.add_image('depth_mono', np.transpose(depth_mono_np, (2, 0, 1)), total_step, dataformats='CHW')
                    else:
                        print("No tensorboard writer found")
                else:
                    print("No tensorboard tracker found")

            if (total_step > 0) and (total_step % cfg.save_frequency == 0):
                if accelerator.is_main_process:
                    save_path = Path(cfg.save_path + '/%d.pth' % (total_step))
                    print(f"Saving checkpoint to {save_path}")
                    model_save = accelerator.unwrap_model(model)
                    torch.save(model_save.state_dict(), save_path)
                    del model_save
                    print(f"Saved checkpoint to {save_path} successfully")
                else:
                    print("No main process found for saving checkpoint")
        

                model.train()
                model.module.freeze_bn()

            if total_step == cfg.total_step:
                should_keep_training = False
                break

    if accelerator.is_main_process:
        save_path = Path(cfg.save_path + '/final.pth')
        model_save = accelerator.unwrap_model(model)
        torch.save(model_save.state_dict(), save_path)
        del model_save
    
    accelerator.end_training()

if __name__ == '__main__':
    main()