# Standard Libraries
import os
import cv2
import time
import argparse
from typing import Tuple
import warnings

# Third-Party Libraries
import numpy as np
from tqdm import tqdm

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

# My Libraries
from .model import get_model
from .utils import (
    fast_intersection_and_union,
    fast_intersection_and_union_2,
    setup_seed,
    ensure_dir,
    resume_random_state,
    find_free_port,
    setup,
    cleanup,
    get_cfg,
)
from .model import PSPNet
from .dataset.data import get_val_loader
from .classifier import TransitionClassifier
from .dataset.classes import classId2className, update_novel_classes

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")
    return get_cfg(parser)


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    print(f"==> Running evaluation script")
    setup(args, rank, world_size)
    setup_seed(args.manual_seed)

    # ========== Data  ==========
    val_loader = get_val_loader(args)

    # ========== Model  ==========
    print("=> Creating the model")
    model = get_model(args).to(rank)

    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained), args.pretrained
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint, strict=False)
        print("=> Loaded weight '{}'".format(args.pretrained))
    else:
        print("=> Not loading anything")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    # ========== Test  ==========
    validate(args=args, val_loader=val_loader, model=model)
    cleanup()


def validate(
    args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP
) -> Tuple[torch.tensor, torch.tensor]:
    print("\n==> Start testing...", flush=True)
    base_novel_classes = classId2className
    random_state = setup_seed(args.manual_seed, return_old_state=True)
    device = torch.device("cuda:{}".format(dist.get_rank()))
    model.eval()

    c = model.module.bottleneck_dim

    # ========== Perform the runs  ==========

    runtime = 0
    features_s, gt_s = None, None
    with torch.no_grad():
        spprt_imgs, s_label = (
            val_loader.dataset.get_support_set()
        )  # Get the support set
        spprt_imgs = spprt_imgs.to(device, non_blocking=True)
        s_label = s_label.to(device, non_blocking=True)

        # avoid OOM
        features_s = []
        for s_img in spprt_imgs:
            fea_s = model.module.extract_features(s_img.unsqueeze(0)).detach()
            features_s.append(fea_s)
        features_s = torch.cat(features_s, dim=0)
        _, _, h, w = features_s.shape
        features_s = features_s.view(
            (args.num_classes_val, args.shot, c, h, w))  # [4, 5, c, h, w]

        gt_s = s_label.view(
            (args.num_classes_val, args.shot, args.image_size, args.image_size))

    # The order of classes in the following tensors is the same as the order of classifier (novels at last)
    cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
    cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
    cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)

    nb_episodes = len(val_loader)  # The number of images in the query set
    for _ in tqdm(range(nb_episodes), leave=True):
        t0 = time.time()
        with torch.no_grad():
            # val_loader只会返回查询集中的数据
            try:
                loader_output = next(iter_loader)
            except (UnboundLocalError, StopIteration):
                iter_loader = iter(val_loader)
                loader_output = next(iter_loader)

            if len(loader_output) == 3:
                qry_img, q_label, image_name = loader_output
            if len(loader_output) == 2:
                qry_img, image_name = loader_output
                q_label = None

            qry_img = qry_img.to(device, non_blocking=True)
            features_q = (
                model.module.extract_features(qry_img).detach().unsqueeze(1)
            )  # [1, 1, 512, 128, 128]
            if q_label is not None:
                q_label = q_label.to(device, non_blocking=True)
                gt_q = q_label.unsqueeze(1)

        # =========== Initialize the classifier and run the method ===============
        base_weight = model.module.classifier.weight.detach().clone()
        base_weight = base_weight.permute(
            *torch.arange(base_weight.ndim - 1, -1, -1))
        base_bias = model.module.classifier.bias.detach().clone()

        classifier = TransitionClassifier(
            args, base_weight, base_bias, n_tasks=features_q.size(0)
        )
        classifier.init_prototypes(features_s, gt_s)
        classifier.optimize(features_s, features_q, gt_s)

        runtime += time.time() - t0

        # =========== Perform inference and compute metrics ===============
        logits = classifier.get_logits(features_q).detach()
        probas = classifier.get_probas(logits)

        # remove car
        probas[:, :, 8] = 0
        # remove building 2
        probas[:, :, 11] = 0

        # 官方运行测试脚本的时候有query_set的label, 所以下面是官方运行脚本的时候运行的部分
        if q_label is not None:
            if args.save_pred_maps is True:  # Save predictions in '.png' file format
                ensure_dir("results/preds")
                n_task, shots, num_classes, h, w = probas.size()
                H, W = gt_q.size()[-2:]
                if (h, w) != (H, W):
                    probas = F.interpolate(
                        probas.view(n_task * shots, num_classes, h, w),
                        size=(H, W),
                        mode="bilinear",
                        align_corners=True,
                    ).view(n_task, shots, num_classes, H, W)
                pred = probas.argmax(2)  # [n_query, shot, H, W]
                pred = np.array(pred.squeeze().cpu(), np.uint8)

                fname = "".join(image_name)

                ##############  post-processing start  ##############

                cascadepsp_building1_path = os.path.join('post-process/cascadepsp/building_type_1', f'{fname}.png')
                cascadepsp_building1 = cv2.imread(cascadepsp_building1_path, cv2.IMREAD_GRAYSCALE)
                pred[cascadepsp_building1 == 7] = 7

                cascadepsp_building2_path = os.path.join('post-process/cascadepsp/building_type_2', f'{fname}.png')
                cascadepsp_building2 = cv2.imread(cascadepsp_building2_path, cv2.IMREAD_GRAYSCALE)
                pred[cascadepsp_building2 == 11] = 11

                vl_sport_path = os.path.join('post-process/ape/cvpr2024_oem_crop-256-128_thres-0.2_sportfield_instance', f'{fname}.png')
                vl_sport = cv2.imread(vl_sport_path, cv2.IMREAD_GRAYSCALE)
                vl_sport = cv2.resize(vl_sport, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                # pred[vl_sport == 1] = 10
                num_labels, vl_sport_labels, stats, _ = cv2.connectedComponentsWithStats(vl_sport, connectivity=8)
                vl_sport_mask = np.zeros_like(vl_sport)
                pred_sport = (pred == 10).astype('uint8')
                for label in range(1, num_labels):
                    if pred_sport[vl_sport_labels == label].sum() >= 1:
                        vl_sport_mask[vl_sport_labels == label] = 1
                pred[vl_sport_mask == 1] = 10

                vl_water_path = os.path.join('post-process/ape/cvpr2024_oem_ori_thres-0.12_water_instance', f'{fname}.png')
                vl_water = cv2.imread(vl_water_path, cv2.IMREAD_GRAYSCALE)
                vl_water = cv2.resize(vl_water, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                num_labels, vl_water_labels, stats, _ = cv2.connectedComponentsWithStats(vl_water, connectivity=8)
                vl_water_mask = np.zeros_like(vl_water)
                vl_bg_mask = np.zeros_like(vl_water)
                # pred_water = (pred == 6).astype('uint8')
                pred_bg = (pred == 0).astype('uint8')
                for label in range(1, num_labels):
                    # Water may be river (background)
                    if pred_bg[vl_water_labels == label].sum() / stats[label, cv2.CC_STAT_AREA] >= 0.5:
                        vl_bg_mask[vl_water_labels == label] = 1
                    else:
                        vl_water_mask[vl_water_labels == label] = 1
                pred[vl_water_mask == 1] = 6
                pred[vl_bg_mask == 1] = 0

                # save parking space area
                parking_pred = (pred == 9).astype('uint8')

                vl_car_path = os.path.join('post-process/ape/cvpr2024_oem_crop-256-128_thres-0.1_car_instance', f'{fname}.png')
                vl_car = cv2.imread(vl_car_path, cv2.IMREAD_GRAYSCALE)
                vl_car = cv2.resize(vl_car, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                num_labels, vl_car_labels, stats, _ = cv2.connectedComponentsWithStats(vl_car, connectivity=8)
                vl_car_mask = np.zeros_like(vl_car)
                # Cars aren't usually on buildings
                pred_building = ((pred == 7) | (pred == 11)).astype('uint8')
                for label in range(1, num_labels):
                    if pred_building[vl_car_labels == label].sum() / stats[label, cv2.CC_STAT_AREA] >= 0.9:
                        continue
                    else:
                        vl_car_mask[vl_car_labels == label] = 1
                pred[vl_car_mask == 1] = 8

                # If there's no car in the parking space, it's the background
                car_pred = (pred == 8).astype('uint8')
                bg_mask = np.zeros_like(parking_pred)
                num_labels, vl_parking_labels, stats, _ = cv2.connectedComponentsWithStats(parking_pred, connectivity=8)
                for label in range(1, num_labels):
                    if car_pred[vl_parking_labels == label].sum() >= 1:
                        continue
                    else:
                        bg_mask[vl_parking_labels == label] = 1
                pred[bg_mask == 1] = 0

                ##############  post-processing end  ##############

                cv2.imwrite(os.path.join(
                    'results/preds', fname + '.png'), pred)

            intersection, union, target = fast_intersection_and_union_2(
                torch.tensor(pred, device=gt_q.device).to(torch.int64), gt_q, num_classes=12
            )  # [batch_size_val, 1, num_classes]
            intersection, union, target = (
                intersection.squeeze(1).cpu(),
                union.squeeze(1).cpu(),
                target.squeeze(1).cpu(),
            )
            cls_intersection += intersection.sum(0)
            cls_union += union.sum(0)
            cls_target += target.sum(0)
        else:
            if args.save_pred_maps is True:  # Save predictions in '.png' file format
                ensure_dir("results/preds")
                n_task, shots, num_classes, h, w = probas.size()
                if (h, w) != (args.image_size, args.image_size):
                    probas = F.interpolate(
                        probas.view(n_task * shots, num_classes, h, w),
                        size=(args.image_size, args.image_size),
                        mode="bilinear",
                        align_corners=True,
                    ).view(n_task, shots, num_classes, args.image_size, args.image_size)
                pred = probas.argmax(2)  # [n_query, shot, H, W]
                pred = np.array(pred.squeeze().cpu(), np.uint8)
                fname = "".join(image_name)

                ##############  post-processing start  ##############

                cascadepsp_building1_path = os.path.join('post-process/cascadepsp/building_type_1', f'{fname}.png')
                cascadepsp_building1 = cv2.imread(cascadepsp_building1_path, cv2.IMREAD_GRAYSCALE)
                pred[cascadepsp_building1 == 7] = 7

                cascadepsp_building2_path = os.path.join('post-process/cascadepsp/building_type_2', f'{fname}.png')
                cascadepsp_building2 = cv2.imread(cascadepsp_building2_path, cv2.IMREAD_GRAYSCALE)
                pred[cascadepsp_building2 == 11] = 11

                vl_sport_path = os.path.join('post-process/ape/cvpr2024_oem_crop-256-128_thres-0.2_sportfield_instance', f'{fname}.png')
                vl_sport = cv2.imread(vl_sport_path, cv2.IMREAD_GRAYSCALE)
                vl_sport = cv2.resize(vl_sport, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                # pred[vl_sport == 1] = 10
                num_labels, vl_sport_labels, stats, _ = cv2.connectedComponentsWithStats(vl_sport, connectivity=8)
                vl_sport_mask = np.zeros_like(vl_sport)
                pred_sport = (pred == 10).astype('uint8')
                for label in range(1, num_labels):
                    if pred_sport[vl_sport_labels == label].sum() >= 1:
                        vl_sport_mask[vl_sport_labels == label] = 1
                pred[vl_sport_mask == 1] = 10

                vl_water_path = os.path.join('post-process/ape/cvpr2024_oem_ori_thres-0.12_water_instance', f'{fname}.png')
                vl_water = cv2.imread(vl_water_path, cv2.IMREAD_GRAYSCALE)
                vl_water = cv2.resize(vl_water, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                num_labels, vl_water_labels, stats, _ = cv2.connectedComponentsWithStats(vl_water, connectivity=8)
                vl_water_mask = np.zeros_like(vl_water)
                vl_bg_mask = np.zeros_like(vl_water)
                # pred_water = (pred == 6).astype('uint8')
                pred_bg = (pred == 0).astype('uint8')
                for label in range(1, num_labels):
                    # Water may be river (background)
                    if pred_bg[vl_water_labels == label].sum() / stats[label, cv2.CC_STAT_AREA] >= 0.5:
                        vl_bg_mask[vl_water_labels == label] = 1
                    else:
                        vl_water_mask[vl_water_labels == label] = 1
                pred[vl_water_mask == 1] = 6
                pred[vl_bg_mask == 1] = 0

                # save parking space area
                parking_pred = (pred == 9).astype('uint8')

                vl_car_path = os.path.join('post-process/ape/cvpr2024_oem_crop-256-128_thres-0.1_car_instance', f'{fname}.png')
                vl_car = cv2.imread(vl_car_path, cv2.IMREAD_GRAYSCALE)
                vl_car = cv2.resize(vl_car, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
                num_labels, vl_car_labels, stats, _ = cv2.connectedComponentsWithStats(vl_car, connectivity=8)
                vl_car_mask = np.zeros_like(vl_car)
                # Cars aren't usually on buildings
                pred_building = ((pred == 7) | (pred == 11)).astype('uint8')
                for label in range(1, num_labels):
                    if pred_building[vl_car_labels == label].sum() / stats[label, cv2.CC_STAT_AREA] >= 0.9:
                        continue
                    else:
                        vl_car_mask[vl_car_labels == label] = 1
                pred[vl_car_mask == 1] = 8

                # If there's no car in the parking space, it's the background
                car_pred = (pred == 8).astype('uint8')
                bg_mask = np.zeros_like(parking_pred)
                num_labels, vl_parking_labels, stats, _ = cv2.connectedComponentsWithStats(parking_pred, connectivity=8)
                for label in range(1, num_labels):
                    if car_pred[vl_parking_labels == label].sum() >= 1:
                        continue
                    else:
                        bg_mask[vl_parking_labels == label] = 1
                pred[bg_mask == 1] = 0

                ##############  post-processing end  ##############

                cv2.imwrite(os.path.join(
                    "results/preds", fname + ".png"), pred)

    if q_label is None:
        return

    base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
    results = []
    results.append("\nClass IoU Results After Few-Shot Learning")
    results.append("---------------------------------------")

    if args.novel_classes is not None:  # Update novel classnames
        update_novel_classes(base_novel_classes, args.novel_classes)

    for i, class_ in enumerate(val_loader.dataset.all_classes):
        if class_ == 0:
            continue

        IoU = cls_intersection[i] / (cls_union[i] + 1e-10)
        classname = base_novel_classes[class_].capitalize()

        if classname == "":
            classname = "Novel class"
        results.append(f"%d %-25s \t %.2f" % (i, classname, IoU * 100))

        if class_ in val_loader.dataset.base_class_list:
            sum_base_IoU += IoU
            base_count += 1
        elif class_ in val_loader.dataset.novel_class_list:
            sum_novel_IoU += IoU
            novel_count += 1

    base_mIoU, novel_mIoU = sum_base_IoU / base_count, sum_novel_IoU / novel_count
    agg_mIoU = (base_mIoU + novel_mIoU) / 2
    wght_base_mIoU, wght_novel_mIoU = base_mIoU * 0.4, novel_mIoU * 0.6
    wght_sum_mIoU = wght_base_mIoU + wght_novel_mIoU

    results.append("---------------------------------------")
    results.append(f"\n%-30s \t %.2f" % ("Base mIoU", base_mIoU * 100))
    results.append(f"%-30s \t %.2f" % ("Novel mIoU", novel_mIoU * 100))
    results.append(
        f"%-30s \t %.2f" % ("Average of Base-and-Novel mIoU", agg_mIoU * 100)
    )
    results.append(f"\n%-30s \t %.2f" %
                   ("Weighted Base mIoU", wght_base_mIoU * 100))
    results.append(f"%-30s \t %.2f" %
                   ("Weighted Novel mIoU", wght_novel_mIoU * 100))
    results.append(
        f"%-30s \t %.2f" % ("Weighted-sum of Base-and-Novel mIoU",
                            wght_sum_mIoU * 100)
    )
    results.append(
        f"The weighted scores are calculated using `0.4:0.6 => base:novel`, which are derived\nfrom the results presented in the SOA GFSS paper adopted as baseline."
    )
    iou_results = "\n".join(results)
    print(iou_results)

    if args.save_ious is True:  # Save class IoUs
        ensure_dir("results")
        with open(os.path.join("results", "base_novel_ious.txt"), "w") as f:
            f.write(iou_results)

    print("\n===> Runtime --- {:.1f}\n".format(runtime))

    resume_random_state(random_state)
    return agg_mIoU


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in args.gpus)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    world_size = len(args.gpus)
    distributed = world_size > 1
    assert not distributed, "Testing should not be done in a distributed way"
    args.distributed = distributed
    args.port = find_free_port()
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    mp.spawn(main_worker, args=(world_size, args),
             nprocs=world_size, join=True)