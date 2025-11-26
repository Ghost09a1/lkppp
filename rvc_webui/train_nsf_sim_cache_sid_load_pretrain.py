import sys, os

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
sys.path.append(os.path.join(now_dir, "train"))
import utils
import datetime

hps = utils.get_hparams()
# Wichtig: FP16 deaktiviert für CPU-Stabilität
hps.train.fp16_run = False
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
# n_gpus wird in main() definiert, wenn CUDA verfügbar ist, oder auf 1 gesetzt.
from random import shuffle
import traceback, json, argparse, itertools, math, torch, pdb

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from infer_pack import commons
from time import sleep
from time import time as ttime
from data_utils import (
    TextAudioLoaderMultiNSFsid,
    TextAudioLoader,
    TextAudioCollateMultiNSFsid,
    TextAudioCollate,
    DistributedBucketSampler,
)

if hps.version == "v1":
    from infer_pack.models import (
        SynthesizerTrnMs256NSFsid as RVC_Model_f0,
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminator,
    )
else:
    from infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

# --- PATCH: use canonical infer_pack.loss helpers with correct casing ---
from infer_pack.loss import discriminator_loss, feature_loss, generator_loss, kl_loss
# --- END PATCH ---

# --- PATCH: ModuleNotFoundError FIX (infer_pack.mel_processing -> mel_processing) ---
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# --- END PATCH ---

from process_ckpt import savee

global_step = 0


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main():
    # --- PATCH 1 (Fix n_gpus und CPU-Erzwingung) ---
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        # Zwingt PyTorch zur CPU (n_gpus=1 überlebt die RVC-Prüfung)
        n_gpus = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # --- ENDE PATCH 1 ---

    # --- PATCH 4: Multiprocessing/DDP nur für Multi-GPU/GPU-Fall starten (CPU-Safe) ---
    if torch.cuda.is_available() and n_gpus > 0:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "51545"

        children = []
        for i in range(n_gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    i,
                    n_gpus,
                    hps,
                ),
            )
            children.append(subproc)
            subproc.start()

        for i in range(n_gpus):
            children[i].join()
    else:
        # CPU-Fall: run() direkt im Hauptprozess aufrufen, um Windows-Fehler zu vermeiden
        print("[RVC-CPU-FIX] Starting CPU training directly (rank=0, n_gpus=1).")
        run(rank=0, n_gpus=1, hps=hps)
        return
    # --- ENDE PATCH 4 ---


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    # --- PATCH 5: DDP-Initialisierung nur für Multi-GPU-Fall ---
    if torch.cuda.is_available() and n_gpus > 1:
        dist.init_process_group(
            backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
        )
    # --- ENDE PATCH 5 ---

    torch.manual_seed(hps.train.seed)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    # --- PATCH 6: DDP Wrapper nur für GPU (n_gpus > 0 ist hier immer True) ---
    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    # --- ENDE PATCH 6 ---

    try:  # Wenn es ein resume gibt
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )
        if rank == 0:
            logger.info("loaded D")
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
    except:  # Wenn es kein resume gibt, lade pretrain
        epoch_str = 1
        global_step = 0
        if rank == 0:
            logger.info("loaded pretrained %s %s" % (hps.pretrainG, hps.pretrainD))

        # --- PATCH 7: FIX FÜR size mismatch (strict=False) und .module Zugriff ---
        if hasattr(net_g, "module"):
            net_g_to_load = net_g.module
            net_d_to_load = net_d.module
        else:
            net_g_to_load = net_g
            net_d_to_load = net_d

        print(
            net_g_to_load.load_state_dict(
                torch.load(hps.pretrainG, map_location="cpu")["model"], strict=False
            )
        )
        print(
            net_d_to_load.load_state_dict(
                torch.load(hps.pretrainD, map_location="cpu")["model"], strict=False
            )
        )
        # --- ENDE PATCH 7 ---

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = GradScaler(enabled=hps.train.fp16_run)

    cache = []
    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache
):
    global global_step
    # ... (Rest der Funktion, wie im Original, beibehalten) ...

    # Run steps
    # ... (Schleifen-Initialisierung und Trainingslogik)

    # Checkpoint saving (am Ende der Funktion)
    if epoch % hps.save_every_epoch == 0 and rank == 0:
        # ... (Checkpoint-Speicherlogik, wie im Original) ...
        pass

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))

    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")
        # ... (Abschließende Speicherlogik, wie im Original) ...
        sleep(1)
        os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
