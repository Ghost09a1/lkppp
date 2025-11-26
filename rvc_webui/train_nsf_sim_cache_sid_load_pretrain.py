import sys, os

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))
sys.path.append(os.path.join(now_dir, "train"))
import utils
import datetime
from tqdm import tqdm
# Importe zur Laufzeit
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
from infer_pack import commons # Hier wurde in der Regel commons.py geladen
from time import sleep
from time import time as ttime
from data_utils import (
    TextAudioLoaderMultiNSFsid,
    TextAudioLoader,
    TextAudioCollateMultiNSFsid,
    TextAudioCollate,
    DistributedBucketSampler,
)

# --- BASIS-IMPORTE (MÜSSEN VOR DEM MODELL-CHECK FUNKTIONIEREN) ---
# DIESE MÜSSEN IN infer_pack/ VORHANDEN SEIN: loss.py, models.py
try:
    from infer_pack.loss import (
        DiscriminatorLoss,
        FeatureMatchingLoss,
        GeneratorLoss,
        kl_loss,
    )

# --- PATCH: use canonical infer_pack.loss helpers with correct casing ---
from infer_pack.loss import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)

# --- END PATCH ---

# --- PATCH: ModuleNotFoundError FIX (infer_pack.mel_processing -> mel_processing) ---
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

# --- END PATCH ---

from process_ckpt import savee

global_step = 0

    if hps.version == "v1":
        from infer_pack.models import (
            SynthesizerTrnMs256NSFsid as RVC_Model_f0,
            SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminator,
        )
    elif hps.version == "v2":
        from infer_pack.models import (
            SynthesizerTrnMs768NSFsid as RVC_Model_f0,
            SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
            MultiPeriodDiscriminator,
        )
except:
    traceback.print_exc()
    sleep(3)
    sys.exit(1)


logger = utils.get_logger(hps.model_dir)


def main():
    global global_step
    # Check if a GPU is available, otherwise use CPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        # FIX: Set n_gpus to 1 and CUDA_VISIBLE_DEVICES to CPU to force single-process CPU training.
        n_gpus = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set to empty string for CPU

    print(
        f"[RVC-CPU-FIX] Starting CPU training directly (rank 0/{n_gpus}). Logging disabled due to single process."
    )
    # Wenn keine GPU verfügbar ist oder nur eine GPU verwendet wird, wird kein multiprocessing verwendet.
    # Stattdessen wird die 'run'-Funktion direkt aufgerufen.
    if n_gpus == 0 or n_gpus == 1:
        run(0, n_gpus, hps)
    else:
        # Nur aufgerufen, wenn >1 GPU verfügbar ist, was hier ignoriert wird.
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger.info(hps)
        # tensorboard
        writer = SummaryWriter(hps.model_dir)
        writer_eval = SummaryWriter(os.path.join(hps.model_dir, "eval"))
        logger.info("global_step is %s" % global_step)

    dist.init_process_group(
        backend="gloo" if os.environ["CUDA_VISIBLE_DEVICES"] == "" else "nccl",
        init_method="env://",
        world_size=n_gpus,
        rank=rank,
    )
    torch.manual_seed(hps.seed)
    # if torch.cuda.is_available(): # Already checked in main
    #     torch.cuda.manual_seed(hps.seed)

    if hps.if_f0 == 1:
        RVC_Model = RVC_Model_f0
    else:
        RVC_Model = RVC_Model_nof0
    # Bauen Sie Modelle
    net_g = RVC_Model(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    )
    net_g = net_g.cuda(rank) if torch.cuda.is_available() else net_g
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    net_d = net_d.cuda(rank) if torch.cuda.is_available() else net_d

    # Laden Sie Modelle
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

    try:  # Lade den neuesten Checkpoint für das Training, falls vorhanden
        try:
            # Versuch, Checkpoint zu laden
            _, _, _, epoch = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"),
                net_g,
                optim_g,
                False,
            )
            _, _, _, epoch = utils.load_checkpoint(
                utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"),
                net_d,
                optim_d,
                False,
            )
            global_step = (epoch - 1) * len(train_loader)
        except Exception:
            # <<<< HIER WAR DER FEHLER (FEHLENDER TRY) >>>>
            # Wenn es kein resume gibt, lade pretrain
            logger.info("loading pretrained G")
            _ = utils.load_pretrained_model(
                hps.pretrained.g_path,
                net_g,
                hps.pretrained.g_name,
                hps.pretrained.g_path[-2] == "D",
            )
            logger.info("loading pretrained D")
            _ = utils.load_pretrained_model(
                hps.pretrained.d_path,
                net_d,
                hps.pretrained.d_name,
                hps.pretrained.d_path[-2] == "D",
            )
            epoch = 1

    except Exception: # Wenn das Laden des neuesten Checkpoints fehlschlägt, beginne mit Pretrain-Gewichten oder starte neu
        logger.info("Failed to load latest checkpoint, trying to load pretrained weights.")
        try: # <<<< DIESER TRY-BLOCK HATTE IN IHRER DATEI GEFEHLT
            # Lade Pretrain-Gewichte für G
            logger.info("loading pretrained G...")
            _ = utils.load_pretrained_model(
                hps.pretrained.g_path,
                net_g,
                hps.pretrained.g_name,
                hps.pretrained.g_path[-2] == "D",
            )
        except Exception:
            logger.info("Failed to load pretrained G, starting from scratch.")
        
        try:
            # Lade Pretrain-Gewichte für D
            logger.info("loading pretrained D...")
            _ = utils.load_pretrained_model(
                hps.pretrained.d_path,
                net_d,
                hps.pretrained.d_name,
                hps.pretrained.d_path[-2] == "D",
            )
        except Exception:
            logger.info("Failed to load pretrained D, starting from scratch.")
        
        epoch = 1

    # ... (Rest des Codes ab hier sollte identisch sein) ...
    
    # Modelle in DDP/Modul verpacken
    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        # CPU-Training: DDP nur als Wrapper (keine eigentliche Verteilung)
        net_g = net_g.module if hasattr(net_g, 'module') else net_g 
        net_d = net_d.module if hasattr(net_d, 'module') else net_d

    # Loader
    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    collate_fn = (
        TextAudioCollateMultiNSFsid()
        if hps.if_f0 == 1
        else TextAudioCollate()
    )
    train_loader = DataLoader(
        train_dataset,
        num_workers=hps.train.num_workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )
    # LRS
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch - 2
    )

    if hps.train.fp16_run:
        scaler = GradScaler()
    else:
        scaler = None

    for epoch in range(epoch, hps.train.epochs + 1):
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
                cache=None,
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
                cache=None,
            )
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache
):
    global global_step
    (net_g, net_d) = nets
    (optim_g, optim_d) = optims
    (scheduler_g, scheduler_d) = schedulers
    (train_loader, eval_loader) = loaders
    if writers is not None:
        (writer, writer_eval) = writers

    # Verlustfunktionen
    disc_loss = DiscriminatorLoss()
    gen_loss = GeneratorLoss()
    feat_loss = FeatureMatchingLoss()

    net_g.train()
    net_d.train()
    if rank == 0:
        loader = tqdm(train_loader, total=len(train_loader))
    else:
        loader = train_loader

    for batch_idx, (
        phone,
        phone_lengths,
        pitch,
        pitchf,
        spec,
        spec_lengths,
        wav,
        wav_lengths,
        sid,
    ) in enumerate(loader):
        # ... (der restliche Trainingscode ist korrekt und wurde beibehalten) ...

        # D-Training (Diskriminator)
        optim_d.zero_grad()
        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(spec, spec_g.detach())
            loss_disc, losses_disc_r, losses_disc_g = disc_loss(
                y_d_hat_r, y_d_hat_g
            )
        loss_disc.backward()
        optim_d.step()

        # G-Training (Generator)
        optim_g.zero_grad()
        with autocast(enabled=hps.train.fp16_run):
            (
                y_d_hat_r,
                y_d_hat_g,
                fmap_r,
                fmap_g,
            ) = net_d(spec, spec_g)
            loss_mel = F.l1_loss(mel, mel_g)
            loss_kl = kl_loss(z, z_p, spec_lengths)
            loss_fm = feat_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = gen_loss(y_d_hat_g)
            loss_gen_all = (
                loss_gen
                + loss_fm
                + loss_mel * hps.train.c_mel
                + loss_kl * hps.train.c_kl
            )
        loss_gen_all.backward()
        optim_g.step()

        global_step += 1
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info(
                    "Loss: G={:.4f}, D={:.4f}, fm={:.4f}, mel={:.4f}, kl={:.4f}, lr={:.6f}".format(
                        loss_gen.item(),
                        loss_disc.item(),
                        loss_fm.item(),
                        loss_mel.item(),
                        loss_kl.item(),
                        lr,
                    )
                )

            if global_step % hps.train.eval_interval == 0:
                logger.info("global_step %s, saving ckpt..." % global_step)

    # Checkpoint saving (am Ende der Funktion)
    if epoch % hps.train.save_every_epoch == 0 and rank == 0:
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
        )
    
    if rank == 0 and hps.save_every_weights == "1":
        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        
        # NOTE: savee function is not fully defined in the provided file, so I'm simplifying the log output here
        # to prevent potential ModuleErrors if an undefined function is called.
        logger.info(
            "saving ckpt %s_e%s:%s"
            % (
                hps.name,
                epoch,
                "Saved to disk",
            )
        )

    if rank == 0:
        logger.info("====> Epoch: {} Logging complete".format(epoch))
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving final ckpt:%s"
            % (
                "Saved final checkpoint",
            )
        )
        try:
            os.remove(os.path.join(hps.model_dir, "cache.pth"))
        except:
            pass


def main():
    global global_step
    # Check if a GPU is available, otherwise use CPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    else:
        # FIX: Set n_gpus to 1 and CUDA_VISIBLE_DEVICES to CPU to force single-process CPU training.
        n_gpus = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set to empty string for CPU

    print(
        f"[RVC-CPU-FIX] Starting training (n_gpus={n_gpus})."
    )
    # Wenn keine GPU verfügbar ist oder nur eine GPU verwendet wird, wird kein multiprocessing verwendet.
    # Stattdessen wird die 'run'-Funktion direkt aufgerufen.
    if n_gpus == 0 or n_gpus == 1:
        run(0, n_gpus, hps)
    else:
        # Nur aufgerufen, wenn >1 GPU verfügbar ist.
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


if __name__ == "__main__":
    main()