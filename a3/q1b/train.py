import torch
from torch import nn
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader
from shapely.geometry import Polygon

import cv2
import matplotlib.pyplot as plt
import wandb

from detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(data):
    return tuple(zip(*data))


def get_iou(det, gt):

    det_x1, det_y1, det_x2, det_y2, det_theta = det
    gt_x1, gt_y1, gt_x2, gt_y2, gt_theta = gt

    def create_polygon(bbox):
        x1, y1, x2, y2, theta = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        angle_rad = np.deg2rad(theta)

        rect_vertices = [
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2)
        ]

        rotated_vertices = []
        for x, y in rect_vertices:
            rotated_x = x * np.cos(angle_rad) - y * np.sin(angle_rad) + center_x
            rotated_y = x * np.sin(angle_rad) + y * np.cos(angle_rad) + center_y
            rotated_vertices.append((rotated_x, rotated_y))
        return Polygon(rotated_vertices)

    det_polygon = create_polygon(det)
    gt_polygon = create_polygon(gt)
    area_intersection = det_polygon.intersection(gt_polygon).area

    area_union = float(det_polygon.area + gt_polygon.area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='area'):
    # det_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, theta, score], ...],
    #       'car' : [[x1, y1, x2, y2, theta, score], ...]
    #   }
    #   {det_boxes_img_2},
    #   ...
    #   {det_boxes_img_N},
    # ]
    #
    # gt_boxes = [
    #   {
    #       'person' : [[x1, y1, x2, y2, theta], ...],
    #       'car' : [[x1, y1, x2, y2, theta], ...],
    #   },
    #   {gt_boxes_img_2},
    #   ...
    #   {gt_boxes_img_N},
    # ]

    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    # average precisions for ALL classes
    aps = []
    best_precisions = []
    best_recalls = []
    for idx, label in enumerate(gt_labels):
        # Get detection predictions of this class
        cls_dets = [
            [im_idx, im_dets_label] for im_idx, im_dets in enumerate(det_boxes)
            if label in im_dets for im_dets_label in im_dets[label]
        ]

        # cls_dets = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, theta_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, theta_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, theta_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, theta_N, score_N]),
        #   ...
        # ]

        # Sort them by confidence score
        cls_dets = sorted(cls_dets, key=lambda k: -k[1][-1])

        # For tracking which gt boxes of this class have already been matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Number of gt boxes for this class for recall calculation
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)
        # For each prediction
        for det_idx, (im_idx, det_pred) in enumerate(cls_dets):
            # Get gt boxes for this image and this label
            im_gts = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            # Get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(det_pred[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # TP only if iou >= threshold and this gt has not yet been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_idx] = 1
            else:
                tp[det_idx] = 1
                # If tp then we set this gt box as matched
                gt_matched[im_idx][max_iou_gt_idx] = True
        # Cumulative tp and fp
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts, eps)
        precisions = tp / np.maximum((tp + fp), eps)

        if method == 'area':
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))

            # Replace precision values with recall r with maximum precision value
            # of any recall value >= r
            # This computes the precision envelope
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            # For computing area, get points where recall changes value
            i = np.where(recalls[1:] != recalls[:-1])[0]
            # Add the rectangular areas to get ap
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            ap = 0.0
            for interp_pt in np.arange(0, 1 + 1E-3, 0.1):
                # Get precision values for recall values >= interp_pt
                prec_interp_pt = precisions[recalls >= interp_pt]

                # Get max of those precision values
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0.0 else 0.0
                ap += prec_interp_pt
            ap = ap / 11.0
        else:
            raise ValueError('Method can only be area or interp')
        if num_gts > 0:
            aps.append(ap)
            f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, eps)
            best_idx = np.argmax(f1_scores)
            best_precisions.append(precisions[best_idx])
            best_recalls.append(recalls[best_idx])
    # compute mAP at provided iou threshold
    mean_ap = sum(aps) / len(aps)
    mean_precision = sum(best_precisions) / len(best_precisions)
    mean_recall = sum(best_recalls) / len(best_recalls)
    return mean_ap, mean_precision, mean_recall


def draw_oriented_box(image, x1, y1, x2, y2, theta, color, thickness=2):

    cx = float((x1 + x2) / 2.0)
    cy = float((y1 + y2) / 2.0)
    w = float(abs(x2 - x1))
    h = float(abs(y2 - y1))
    box = cv2.boxPoints(((cx, cy), (w, h), float(theta))).astype(int)
    cv2.drawContours(image, [box], 0, color, thickness)
    return image


def visualize_bounding_box(detection, target, image, dest, idx, step):

    image = image[0].detach().cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for jdx in range(len(detection[0]['boxes'])):
        x1, y1, x2, y2, theta = detection[0]['boxes'][jdx].detach().cpu().numpy().astype(int)
        draw_oriented_box(image, x1, y1, x2, y2, theta, color=(255, 0, 0))
    for jdx in range(len(target[0]['boxes'][0])):
        x1, y1, x2, y2, theta = target[0]['boxes'][0][jdx].detach().cpu().numpy().astype(int)
        draw_oriented_box(image, x1, y1, x2, y2, theta, color=(0, 255, 0))
    cv2.imwrite(f'{dest}/boxes_{idx + 1}_{step + 1}.png', image)


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']

    if train_config['wandb_log']:
        wandb.init(project='cv-s25-a3-faster-rcnn', name=train_config['task_name'], reinit=True)

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    st_train = SceneTextDataset('train', dataset_config['root_dir'], train_config['image_augmentation'])
    st_test = SceneTextDataset('test', dataset_config['root_dir'])

    train_dataset = DataLoader(st_train,
                               batch_size=4,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=collate_function)
    test_dataset  = DataLoader(st_test,
                               batch_size=1,
                               shuffle=False)

    faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True,
                                                min_size=600,
                                                max_size=1000
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes']
    )
    faster_rcnn_model.load_state_dict(torch.load('tv_frcnn_r50fpn_faster_rcnn_st.pth'))

    faster_rcnn_model.roi_heads.box_predictor.angle_pred = nn.Linear( \
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        180 // train_config['angle_step_size'] if train_config['angle_prediction'] == 'classification' else 1
    )

    if train_config['resume']:
        model_path = os.path.join(train_config['task_name'], 'tv_frcnn_r50fpn_' + train_config['ckpt_name'])
        faster_rcnn_model.load_state_dict(torch.load(model_path))
        print(f"Loaded {model_path}")

    faster_rcnn_model.train()
    faster_rcnn_model.to(device)
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])

    optimizer = torch.optim.SGD(lr=1E-4,
                                params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
                                weight_decay=5E-5, momentum=0.9)

    num_epochs = train_config['num_epochs']
    step_count = 0

    os.makedirs(train_config['task_name'], exist_ok=True)
    if train_config['visualize']:
        os.makedirs(os.path.join(train_config['task_name'], 'visualization'), exist_ok=True)

    for i in range(num_epochs):
        rpn_classification_losses = []
        rpn_localization_losses = []
        frcnn_classification_losses = []
        frcnn_localization_losses = []
        frcnn_angle_losses = []

        faster_rcnn_model.train()
        for ims, targets, _ in tqdm(train_dataset):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = [im.float().to(device) for im in ims]
            batch_losses = faster_rcnn_model(images, targets)
            loss = batch_losses['loss_classifier']
            loss += batch_losses['loss_box_reg']
            loss += batch_losses['loss_rpn_box_reg']
            loss += batch_losses['loss_objectness']
            loss += batch_losses['loss_angle'] * train_config['angle_loss_weight']

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())
            frcnn_angle_losses.append(batch_losses['loss_angle'].item())

            loss.backward()
            optimizer.step()
            step_count +=1

        torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                'tv_frcnn_r50fpn_' + train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        loss_output += ' | FRCNN Angle Loss : {:.4f}'.format(np.mean(frcnn_angle_losses))
        print(loss_output)

        if train_config['visualize']:
            faster_rcnn_model.eval()
            with torch.no_grad():
                count = 0
                for im, target, im_path in test_dataset:
                    target['boxes'] = target['bboxes'].float().to(device)
                    del target['bboxes']
                    target['labels'] = target['labels'].long().to(device)
                    image = im.float().to(device)
                    detection = faster_rcnn_model(image, None)
                    visualize_bounding_box(detection, [target], image, os.path.join(train_config['task_name'], 'visualization'), count, i)
                    count += 1
                    if count >= 4:
                        break

        faster_rcnn_model.eval()
        with torch.no_grad():
            gts = []
            preds = []
            for im, target, _ in tqdm(test_dataset):
                im = im.float().to(device)
                target_boxes = target['bboxes'].float().to(device)[0]
                target_labels = target['labels'].long().to(device)[0]

                frcnn_output = faster_rcnn_model(im, None)[0]
                boxes = frcnn_output['boxes']
                labels = frcnn_output['labels']
                scores = frcnn_output['scores']

                pred_boxes = {}
                gt_boxes = {}
                for label_name in st_test.label2idx:
                    pred_boxes[label_name] = []
                    gt_boxes[label_name] = []

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2, theta = box.detach().cpu().numpy()
                    label = labels[idx].detach().cpu().item()
                    score = scores[idx].detach().cpu().item()
                    label_name = st_test.idx2label[label]
                    pred_boxes[label_name].append([x1, y1, x2, y2, theta, score])

                for idx, box in enumerate(target_boxes):
                    x1, y1, x2, y2, theta = box.detach().cpu().numpy()
                    label = target_labels[idx].detach().cpu().item()
                    label_name = st_test.idx2label[label]
                    gt_boxes[label_name].append([x1, y1, x2, y2, theta])

                gts.append(gt_boxes)
                preds.append(pred_boxes)

            mean_ap, precision, recall = compute_map(preds, gts, method='interp')
            metric_output = ''
            metric_output += 'mAP : {:.4f}'.format(mean_ap)
            metric_output += ' | Precision : {:.4f}'.format(precision)
            metric_output += ' | Recall : {:.4f}'.format(recall)
            print(metric_output)
            print('Finished epoch {}'.format(i))

        if train_config['wandb_log']:
            wandb.log({
                "RPN Classification Loss": np.mean(rpn_classification_losses),
                "RPN Localization Loss": np.mean(rpn_localization_losses),
                "FRCNN Classification Loss": np.mean(frcnn_classification_losses),
                "FRCNN Localization Loss": np.mean(frcnn_localization_losses),
                "FRCNN Angle Loss": np.mean(frcnn_angle_losses),
                "Mean Average Precision": mean_ap,
                "Precision": precision,
                "Recall": recall
            })

    faster_rcnn_model.eval()
    with torch.no_grad():
        gts = []
        preds = []
        for im, target, _ in tqdm(test_dataset):
            im = im.float().to(device)
            target_boxes = target['bboxes'].float().to(device)[0]
            target_labels = target['labels'].long().to(device)[0]

            frcnn_output = faster_rcnn_model(im, None)[0]
            boxes = frcnn_output['boxes']
            labels = frcnn_output['labels']
            scores = frcnn_output['scores']

            pred_boxes = {}
            gt_boxes = {}
            for label_name in st_test.label2idx:
                pred_boxes[label_name] = []
                gt_boxes[label_name] = []

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2, theta = box.detach().cpu().numpy()
                label = labels[idx].detach().cpu().item()
                score = scores[idx].detach().cpu().item()
                label_name = st_test.idx2label[label]
                pred_boxes[label_name].append([x1, y1, x2, y2, theta, score])

            for idx, box in enumerate(target_boxes):
                x1, y1, x2, y2, theta = box.detach().cpu().numpy()
                label = target_labels[idx].detach().cpu().item()
                label_name = st_test.idx2label[label]
                gt_boxes[label_name].append([x1, y1, x2, y2, theta])

            gts.append(gt_boxes)
            preds.append(pred_boxes)

        mean_ap = {}
        for iou_threshold in np.arange(0.50, 1.00, 0.05):
            mean_ap[f'mAP@{iou_threshold:.2f}'] = compute_map(preds, gts, iou_threshold=iou_threshold, method='interp')[0]

        print(mean_ap)
        if train_config['wandb_log']:
            wandb.log(mean_ap)

    print('Done Training...')
    if train_config['wandb_log']:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/st.yaml', type=str)
    args = parser.parse_args()
    train(args)