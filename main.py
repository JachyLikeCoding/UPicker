import argparse
import datetime
import json
import random
import time
import numpy as np
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, viz
from models import build_model
from models.backbone import build_swav_backbone, build_swav_backbone_old
import util.misc as utils
from util.default_args import set_model_defaults, get_args_parser
from util.pytorchtools import EarlyStopping
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from util.utils import BestMetricHolder, ModelEma

try:
    # from torch.utils.tensorboard import SummaryWriter
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def main(args):
    utils.init_distributed_mode(args)
    print("git: \n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only."
    
    print(args)
    tmp = args.output_dir.split("/")[-1]
    print('tmp: ',tmp)

    log_dir = f'runs/upicker_experiment_{tmp}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    use_cuda = torch.cuda.is_available()
    print('Use cuda {}'.format(use_cuda))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    if args.random_seed:
        args.seed = np.random.randint(0, 1000000)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create Tensorboard writer
    if TENSORBOARD_FOUND:
        writer = SummaryWriter(log_dir)
    else:
        print("Tensorboard not available: not logging progress")

    swav_model = None
    if args.dataset.endswith('pretrain'):
        if args.obj_embedding_head == 'head':
            swav_model = build_swav_backbone(args, device)
        elif args.obj_embedding_head == 'intermediate':
            swav_model = build_swav_backbone_old(args, device)

    if args.model == 'upicker':
        print('\n[args by parser:]', args)
        from util.slconfig import SLConfig, DictAction
        cfg = SLConfig.fromfile(args.config_file)
        print('\n[upicker args by file:]', cfg)
        
        cfg_dict = cfg._cfg_dict.to_dict()
        for k,v in cfg_dict.items():
            setattr(args, k, v)

        if args.options is not None:
            cfg.merge_from_dict(args.options)
        print('args after merge:\n', args)

        from models.upicker import upicker
        model, criterion, postprocessors = upicker.build_upicker(args)
    else:
        model, criterion, postprocessors = build_model(args)
        
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # if args.model != 'upicker':
    #     tensor = torch.rand(1, 3, 800, 800)
    #     flops = FlopCountAnalysis(model, tensor.to(device))
    #     print("FLOPs: ", flops.total())

    # 分析parameters
    print(parameter_count_table(model))

    dataset_train, dataset_val = get_datasets(args)
    
    print("Number of training examples:", len(dataset_train))
    print("Number of validation examples:", len(dataset_val))

    # 数据集采样器
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    coco_evaluator = None

    # batch采样器
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    # dataloader
    data_loader_train = DataLoader(dataset_train,
                                    batch_sampler=batch_sampler_train,
                                    collate_fn=utils.collate_fn,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
    data_loader_val = DataLoader(dataset_val,
                                    args.batch_size,
                                    sampler=sampler_val,
                                    drop_last=False,
                                    collate_fn=utils.collate_fn,
                                    num_workers=args.num_workers,
                                    pin_memory=True)


    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out
    
    for n, p in model_without_ddp.named_parameters():
        print('n:',n)
    
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names)
                            and not match_name_keywords(n, args.lr_linear_proj_names)
                            and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, args.lr_backbone_names)
                            and p.requires_grad],
            "lr": args.lr_backbone,
        }, 
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                        if match_name_keywords(n, args.lr_linear_proj_names)
                            and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]

    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        # default
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.pretrain:
        print("Initialized from the pre-training model")
        checkpoint = torch.load(args.pretrain, map_location='cpu')
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            # remove useless class embedding
            if 'class_embed' in k:
                del state_dict[k]
        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print('msg: ',msg)
    
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (
            k.endswith('total_params') or k.endswith('total_ops'))]
        
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            import copy
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            print(optimizer.param_groups)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(
                    map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

        if (not args.eval and not args.viz):
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args=args
            )

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors, 
                                            data_loader_val, base_ds, device, args.output_dir, args=args, writer=writer)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir/"eval.pth")
        return
    
    if args.viz:
        viz(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir)
        return

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=30, verbose=True)
    
    print("\n>>>>>>>>>>>>>> UPicker Start training >>>>>>>>>>>>>>>>>>")
    print('Start epoch: ', args.start_epoch)
    print('Epochs', args.epochs)
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    
    for epoch in range(args.start_epoch, args.epochs):
        print("====== This is the training epoch: " + str(epoch) + "======")
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(model, swav_model, criterion, data_loader_train, optimizer, writer, 
                                      device, epoch, args.clip_max_norm, args=args)
        
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir/'checkpoint.pth']
            # extra checkpoints before LR drop and every 10 epochs
            if (epoch+1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                # checkpoint_paths.append(output_dir/f'checkpoint{epoch:04}.pth')
                checkpoint_paths.append(output_dir/'checkpoint{:04}.pth'.format(epoch))
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch':epoch,
                    'args':args,
                }, checkpoint_path)
        
        # EarlyStopping
        if args.dataset in ['coco'] and epoch % args.eval_every == 0:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args, writer)
            # early_stopping(test_stats['loss'], model)
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = output_dir/'checkpoint_best_regular.pth'
                utils.save_on_master({
                    'model':model_without_ddp.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
                
            
            pr1 = coco_evaluator.coco_eval['bbox'].eval['precision'][0,:,0,:,2]
            x = np.arange(0.0, 1.01, 0.01)

            plt.switch_backend('agg')
            plt.figure(figsize=(10,6), facecolor='w')
            
            plt.xlabel('Recall', fontsize=14)
            plt.ylabel('Precision', fontsize=14)
            plt.xlim(0,1.0)
            plt.ylim(0,1.0)

            plt.grid(True, linestyle='--', alpha=0.7)

            plt.plot(x, pr1, 'r-', label='Precision-Recall curve', linewidth=2)
            handles, labels = plt.gca().get_legend_handles_labels()

            from collections import OrderedDict
            by_label = OrderedDict(zip(labels, handles))

            plt.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=12)
            plt.savefig(f'{args.output_dir}/PR.png',dpi=300, bbox_inches='tight')
            plt.close()

        else:
            test_stats = {}
        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch':epoch,
                    'n_parameters':n_parameters}
        
        for k, v in train_stats.items():
            writer.add_scalar(k, v, epoch)

        if args.output_dir and utils.is_main_process():
            with (output_dir/'log.txt').open('a') as f:
                f.write(json.dumps(log_stats)+'\n')
            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir/'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 10 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir/"eval"/name)

        if early_stopping.early_stop:
            print(f'Early stopping......after {epoch} epochs')
            break
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('\nTraining time {}'.format(total_time_str))
    writer.close()


def get_datasets(args):
    print('args.dataset == ', args.dataset)
    # if we want to pretrain on the dataset：
    if args.dataset.endswith('pretrain'):
        from datasets.selfdet import build_selfdet
        if args.filter_num > 0:
            tmp = os.path.join(os.path.join(args.data_root, args.dataset_file), f'pretrain_{args.filter_num}')
        else:
            tmp = os.path.join(os.path.join(args.data_root, args.dataset_file), 'pretrain')
        dataset_train = build_selfdet('pretrain', args=args, p=tmp)
        dataset_val = build_dataset(image_set='val', args=args)
    # if we want to finetune on this dataset:
    else:
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)

    return dataset_train, dataset_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    set_model_defaults(args)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)