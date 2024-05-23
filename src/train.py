import random
import numpy as np
import torch
import torch.nn as nn

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

import os
import argparse
import logging
import mlflow
import datetime
import scipy
from sklearn.metrics import roc_auc_score
from torch import optim
from tqdm.auto import tqdm
from data.dataloader import build_random_train_val_dataset
from configs.base import Config, import_config
from models import networks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main(cfg: Config):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.model_type)
    if cfg.model_type == "BirdClassification":
        cfg.checkpoint_dir = os.path.join(
            cfg.checkpoint_dir, cfg.audio_encoder_type, current_time
        )
    else:
        cfg.checkpoint_dir = os.path.join(
            cfg.checkpoint_dir, cfg.panns_type, current_time
        )

    # Log, weight, mlflow folder
    log_dir = os.path.join(cfg.checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    ## Add logger to log folder
    logging.getLogger().setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(cfg.name)
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    ## Add mlflow to log folder
    mlflow.set_tracking_uri(
        uri=f'file://{os.path.abspath(os.path.join(log_dir, "mlruns"))}'
    )

    # Preparing checkpoint output
    weight_best_path = os.path.join(cfg.checkpoint_dir, "weight_best.pth")
    weight_last_path = os.path.join(cfg.checkpoint_dir, "weight_last.pt")

    # Build dataset
    logger.info("Building dataset...")
    (train_dataloader, test_dataloader), num_classes = build_random_train_val_dataset(
        data_root=cfg.data_root,
        data_type=cfg.data_type,
        batch_size=cfg.batch_size,
        seed=SEED,
        max_audio_sec=cfg.max_audio_sec,
    )
    cfg.num_classes = num_classes

    # Save configs
    logger.info("Saving config to {}".format(cfg.checkpoint_dir))
    cfg.save(cfg.checkpoint_dir)
    cfg.show()

    logger.info("Building model, loss and optimizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = getattr(networks, cfg.model_type)(cfg)
    model.to(device)

    ccl = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.lr_step_size,
        gamma=cfg.gamma,
    )

    best_loss = float(np.inf)
    best_auc = -1.0

    global_train_step = 0
    global_val_step = 0
    epoch_current = -1
    epoch_current_step = -1
    num_steps = len(train_dataloader)

    resume = cfg.resume
    if resume:
        checkpoint = torch.load(weight_last_path)
        epoch_current = checkpoint["epoch"]
        epoch_current_step = checkpoint["epoch_current_step"]
        global_train_step = checkpoint["global_train_step"]
        global_val_step = checkpoint["global_val_step"]
        best_loss = checkpoint["best_loss"]
        model.load_state_dict(checkpoint["state_dict_model"])
        optimizer.load_state_dict(checkpoint["state_dict_optim_model"])
        lr_scheduler.load_state_dict(checkpoint["state_dict_scheduler_model"])
        logger.info(
            "Resume training from Epoch {}/{} - Step {}/{}".format(
                epoch_current + 1, cfg.epochs, epoch_current_step, num_steps
            )
        )
    else:
        logger.info("Start training...")

    with mlflow.start_run():
        for epoch in range(cfg.epochs):
            if epoch < 5:
                model.freeze_encoder()
            else:
                model.unfreeze_encoder()

            if epoch < epoch_current:
                continue
            total_loss_train = []
            total_auc_train = []
            logger.info("Train epoch {}/{}".format(epoch + 1, cfg.epochs))

            model.train()
            with tqdm(total=num_steps, ascii=True) as pbar:
                for step, (inputs, labels) in enumerate(iter(train_dataloader)):
                    if cfg.resume:
                        if epoch_current_step < step:
                            continue
                        else:
                            resume = False
                            continue
                    global_train_step += 1

                    optimizer.zero_grad()

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = ccl(outputs, labels)
                    loss.backward()

                    optimizer.step()

                    inputs = inputs.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    outputs = outputs.detach().cpu().numpy()
                    loss = loss.detach().cpu().numpy()

                    probs = scipy.special.softmax(outputs, axis=1)
                    auc = float(
                        roc_auc_score(
                            labels,
                            probs,
                            multi_class="ovo",
                            average="macro",
                            labels=range(cfg.num_classes),
                        )
                    )

                    total_auc_train.append(auc)
                    total_loss_train.append(loss.item())

                    mlflow.log_metric("loss", loss.item(), step=global_train_step)

                    mlflow.log_metric("auc", auc, step=global_train_step)

                    postfix = "Epoch {}/{} - loss: {:.4f} - auc: {:.4f}".format(
                        epoch + 1, cfg.epochs, loss.item(), auc
                    )
                    pbar.set_description(postfix)
                    pbar.update(1)

                    if global_train_step % cfg.ckpt_save_fred == 0:
                        checkpoint = {
                            "epoch": epoch,
                            "epoch_current_step": step,
                            "global_train_step": global_train_step,
                            "global_val_step": global_val_step,
                            "best_loss": best_loss,
                            "state_dict_model": model.state_dict(),
                            "state_dict_optim_model": optimizer.state_dict(),
                            "state_dict_scheduler_model": lr_scheduler.state_dict(),
                        }
                        torch.save(checkpoint, weight_last_path)

            logger.info(
                "Epoch {}/{} - epoch_loss: {:.4f} - epoch_auc: {:.4f}".format(
                    epoch + 1,
                    cfg.epochs,
                    np.mean(total_loss_train).item(),
                    np.mean(total_auc_train).item(),
                )
            )

            total_loss_val = []
            total_auc_val = []
            model.eval()
            global_val_step += 1
            for inputs, labels in tqdm(iter(test_dataloader)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                    loss = ccl(outputs, labels)

                    inputs = inputs.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    outputs = outputs.detach().cpu().numpy()
                    loss = loss.detach().cpu().numpy()

                    probs = scipy.special.softmax(outputs, axis=1)
                    auc = roc_auc_score(
                        labels,
                        probs,
                        multi_class="ovo",
                        average="macro",
                        labels=range(cfg.num_classes),
                    )

                total_auc_val.append(auc)
                total_loss_val.append(loss.item())
            total_loss = np.mean(total_loss_val).item()
            total_auc = np.mean(total_auc_val).item()

            mlflow.log_metric("val_loss", float(total_loss), step=global_val_step)
            mlflow.log_metric("val_auc", float(total_auc), step=global_val_step)
            logger.info(
                "Epoch {}/{} - val_loss: {:.4f} - val_auc: {:.4f}".format(
                    epoch + 1, cfg.epochs, total_loss, total_auc
                )
            )

            lr_scheduler.step()

            if total_auc > best_auc:
                best_auc = total_auc
                torch.save(model.state_dict(), weight_best_path)

    end_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info("Training finished at {}".format(end_time))


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="../src/configs/base.py",
        help="Path to config.py file",
    )
    parser.add_argument(
        "-rcp",
        "--resume_config_path",
        type=str,
        default=None,
        help="Path to resume cfg.log file if want to resume training",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    cfg: Config = import_config(args.config)
    if cfg.resume and cfg.opt_path:
        resume = cfg.resume
        resume_path = cfg.resume_path
        cfg.load(cfg.opt_path)
        cfg.resume = resume
        cfg.resume_path = resume_path

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main(cfg)
