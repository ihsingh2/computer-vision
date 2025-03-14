import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
from dataset.st import SceneTextDataset
from torch.utils.data.dataloader import DataLoader

import cv2
import matplotlib.pyplot as plt
import wandb

from detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_function(data):
    return tuple(zip(*data))


def visualize_objectness_map(objectness, dest, step):

    for idx in range(len(objectness[0])):
        plt.figure(figsize=(len(objectness) * 5, 4))
        for level in range(len(objectness)):
            obj_map = objectness[level][idx].detach().cpu().permute(1, 2, 0).mean(2).numpy()
            plt.subplot(1, len(objectness), level + 1)
            im = plt.imshow(obj_map, cmap='viridis', vmin=0, vmax=1)
            plt.colorbar(im, shrink=0.7)
            plt.title(f'Level {level + 1}')
        plt.suptitle(f'Objectness Maps for Image {idx + 1}')
        plt.savefig(f'{dest}/objectness_{idx + 1}_{step + 1}.png', bbox_inches='tight')
        plt.close()
        plt.clf()


def visualize_proposals(proposals, images, dest, step):

    for idx in range(len(images)):
        image = images[idx].detach().cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for jdx, box in enumerate(proposals[idx]):
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)
        cv2.imwrite(f'{dest}/proposals_{idx + 1}_{step + 1}.png', image)


def visualize_anchors(anchors, images, dest, step):

    for idx in range(len(images)):
        image = images[idx].detach().cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for jdx, box in enumerate(anchors[idx]):
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            if jdx < 10:
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
        cv2.imwrite(f'{dest}/anchors_{idx + 1}_{step + 1}.png', image)


def visualize_bounding_box(detections, images, dest, step):

    for idx in range(len(images)):
        image = images[idx].detach().cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for jdx in range(len(detections[idx]['boxes'])):
            x1, y1, x2, y2 = detections[idx]['proposals'][jdx].detach().cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=1)
            x1, y1, x2, y2 = detections[idx]['boxes'][jdx].detach().cpu().numpy().astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)
            (text_w, text_h), _ = cv2.getTextSize(f'{detections[idx]['scores'][jdx]:.2f}', cv2.FONT_HERSHEY_PLAIN, 0.65, 1)
            cv2.rectangle(image, (x1, y1), (x1 + text_w + 5, y1 + text_h + 5), (0, 255, 0), -1)
            cv2.putText(image, f'{detections[idx]['scores'][jdx]:.2f}', (x1 + 3, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
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

    st_train = SceneTextDataset('train', root_dir=dataset_config['root_dir'])
    st_test = SceneTextDataset('test', root_dir=dataset_config['root_dir'])

    train_dataset = DataLoader(st_train,
                               batch_size=4,
                               shuffle=True,
                               num_workers=4,
                               collate_fn=collate_function)
    test_dataset  = DataLoader(st_test,
                               batch_size=4,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=collate_function)

    faster_rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True,
                                                min_size=600,
                                                max_size=1000
    )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=dataset_config['num_classes'])

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

            rpn_classification_losses.append(batch_losses['loss_objectness'].item())
            rpn_localization_losses.append(batch_losses['loss_rpn_box_reg'].item())
            frcnn_classification_losses.append(batch_losses['loss_classifier'].item())
            frcnn_localization_losses.append(batch_losses['loss_box_reg'].item())

            loss.backward()
            optimizer.step()
            step_count +=1

        print('Finished epoch {}'.format(i))
        torch.save(faster_rcnn_model.state_dict(), os.path.join(train_config['task_name'],
                                                                'tv_frcnn_r50fpn_' + train_config['ckpt_name']))
        loss_output = ''
        loss_output += 'RPN Classification Loss : {:.4f}'.format(np.mean(rpn_classification_losses))
        loss_output += ' | RPN Localization Loss : {:.4f}'.format(np.mean(rpn_localization_losses))
        loss_output += ' | FRCNN Classification Loss : {:.4f}'.format(np.mean(frcnn_classification_losses))
        loss_output += ' | FRCNN Localization Loss : {:.4f}'.format(np.mean(frcnn_localization_losses))
        print(loss_output)

        if train_config['wandb_log']:
            wandb.log({
                "RPN Classification Loss": np.mean(rpn_classification_losses),
                "RPN Localization Loss": np.mean(rpn_localization_losses),
                "FRCNN Classification Loss": np.mean(frcnn_classification_losses),
                "FRCNN Localization Loss": np.mean(frcnn_localization_losses)
            })

        if train_config['visualize']:
            faster_rcnn_model.eval()
            with torch.no_grad():
                for ims, targets, _ in test_dataset:
                    for target in targets:
                        target['boxes'] = target['bboxes'].float().to(device)
                        del target['bboxes']
                        target['labels'] = target['labels'].long().to(device)
                    images = [im.float().to(device) for im in ims]
                    objectness, proposals, anchors, detections = faster_rcnn_model(images, targets, visualize=True)
                    visualize_objectness_map(objectness, os.path.join(train_config['task_name'], 'visualization'), i)
                    visualize_proposals(proposals, images, os.path.join(train_config['task_name'], 'visualization'), i)
                    visualize_anchors(anchors, images, os.path.join(train_config['task_name'], 'visualization'), i)
                    visualize_bounding_box(detections, images, os.path.join(train_config['task_name'], 'visualization'), i)
                    break

    print('Done Training...')
    if train_config['wandb_log']:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for faster rcnn using torchvision code training')
    parser.add_argument('--config', dest='config_path',
                        default='config/st.yaml', type=str)
    args = parser.parse_args()
    train(args)