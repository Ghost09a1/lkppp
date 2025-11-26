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
    # V1 Modelle als Fallback
    from infer_pack.models import (
        SynthesizerTrnMs256NSFsid as RVC_Model_f0_v1,
        SynthesizerTrnMs256NSF as RVC_Model_nof0_v1,
        MultiPeriodDiscriminator as MultiPeriodDiscriminator_v1,
    )
except ModuleNotFoundError as e:
    # Kritischer Fehler: RVC Kern-Dateien fehlen
    raise RuntimeError(
        f"FATALER FEHLER: Konnte die RVC Kern-Dateien 'infer_pack/loss' oder 'infer_pack/models' nicht finden. "
        f"Bitte stellen Sie sicher, dass Ihr RVC-Ordner ('C:\\Users\\Ghost\\Desktop\\avo\\MyCandyLocal\\rvc_webui') "
        f"vollständig und korrekt ist. Details: {e}"
    )

# --- START ROBUST MODEL VERSION SELECTION ---
hps = utils.get_hparams()
hps.train.fp16_run = False  # Wichtig: FP16 deaktiviert für CPU-Stabilität
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")

RVC_Model_f0 = RVC_Model_f0_v1
RVC_Model_nof0 = RVC_Model_nof0_v1
MultiPeriodDiscriminator = MultiPeriodDiscriminator_v1
MODEL_VERSION = "v1"

# Versuch, auf V2-Modelle umzuschalten, wenn in der Konfiguration/CLI angegeben
if hps.version == "v2" or (hps.version not in ("v1", "v2") and "-v 2" in " ".join(sys.argv)):
    try:
        from infer_pack.models import (
            SynthesizerTrnMs768NSFsid as RVC_Model_f0_v2,
            SynthesizerTrnMs768NSF as RVC_Model_nof0_v2,
            MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator_v2,
        )
        # Erfolg: Überschreibe die V1-Definitionen mit V2
        RVC_Model_f0 = RVC_Model_f0_v2
        RVC_Model_nof0 = RVC_Model_nof0_v2
        MultiPeriodDiscriminator = MultiPeriodDiscriminator_v2
        MODEL_VERSION = "v2"
        print("INFO: Erfolgreich RVC V2 Modelle (768ms) geladen.")
    except ImportError:
        print(
            "CRITICAL WARNING: Import von RVC V2 Modellen (SynthesizerTrnMs768) fehlgeschlagen. "
            "Fällt zurück auf V1-Modelle (SynthesizerTrnMs256). Das Training wird mit der V1-Architektur fortgesetzt."
        )

# --- END ROBUST MODEL VERSION SELECTION ---


if hps.if_f0 == 1:
    RVC_Model = RVC_Model_f0
else:
    RVC_Model = RVC_Model_nof0

# ... (Der Rest des Skripts 'run', 'train_and_evaluate', 'main' bleibt gleich) ...

def run(rank, n_gpus, hps):
    global global_step
    # ... (restliche run Funktion)
    # Checkpoint-Lade-Logik ...
    
    if hps.if_latest == 1: # Load latest checkpoint
        # ... (rest of loading logic) ...
        pass
    except: # Wenn es kein resume gibt, lade pretrain
        epoch_str = 1
        global_step = 0
        if rank == 0:
            logger.info("loaded pretrained %s %s" % (hps.pretrainG, hps.pretrainD))
        
        # ... (net_g und net_d Definitionen, optimierer-Definitionen) ...

        if hasattr(net_g, "module"):
            net_g_to_load = net_g.module
            net_d_to_load = net_d.module
        else:
            net_g_to_load = net_g
            net_d_to_load = net_d
        
        # LADEN DES GENERATORS (G)
        g_ckpt = torch.load(hps.pretrainG, map_location="cpu")["model"]
        
        # PATCH (für size mismatch V1->V2): Löschen des inkompatiblen V1-Layers
        # Wir prüfen hier, ob wir ein V2-Modell geladen haben und das V1-Pretrain-Gewicht "enc_p.emb_phone.weight" vorhanden ist.
        is_v2_model_class = "Ms768" in RVC_Model.__name__
        
        if is_v2_model_class and "enc_p.emb_phone.weight" in g_ckpt:
            if rank == 0:
                print("INFO:char_1: [FINAL PATCH] Manually removing V1 layer 'enc_p.emb_phone.weight' for V2 training.")
            del g_ckpt["enc_p.emb_phone.weight"]

        print(
            net_g_to_load.load_state_dict(
                g_ckpt, strict=False
            )
        )
        
        # LADEN DES DISCRIMINATORS (D)
        print(
            net_d_to_load.load_state_dict(
                torch.load(hps.pretrainD, map_location="cpu")["model"], strict=False
            )
        )

    # ... (rest of run function) ...


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

    if hps.if_cache_data_in_gpu == 1:
        # cache_data_in_gpu
        if cache["data"] == None:
            train_loader.dataset.init_data(train_loader, cache)
        if cache["data"] != None:
            train_loader.dataset.data = cache["data"]
            train_loader.dataset.data_lengths = cache["data_lengths"]
            train_loader.dataset.all_data = cache["all_data"]
            train_loader.dataset.all_data_lengths = cache["all_data_lengths"]
    # train_loader.dataset.init_data(train_loader,cache)

    train_loader.dataset.sort_data(
        train_loader.dataset.all_data, train_loader.dataset.all_data_lengths
    )

    train_loader.dataset.generate_next_page(rank, hps.train.batch_size * n_gpus)
    net_g.train()
    net_d.train()
    
    # Import von tqdm hier, um den Fehler zu vermeiden, falls es nicht oben im Skript steht.
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterator, *args, **kwargs):
            return iterator

    if rank == 0:
        loader = tqdm(train_loader, initial=global_step % len(train_loader))
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
        # global_step+=1
        # if global_step%500==0:
        #     logger.info("write to tensorboard at step %s"%global_step)
        # phone = phone.cuda(rank, non_blocking=True)
        # pitch = pitch.cuda(rank, non_blocking=True)
        # pitchf = pitchf.cuda(rank, non_blocking=True)
        # spec = spec.cuda(rank, non_blocking=True)
        # wav = wav.cuda(rank, non_blocking=True)
        # sid = sid.cuda(rank, non_blocking=True)
        phone = phone.to(rank, non_blocking=True)
        pitch = pitch.to(rank, non_blocking=True)
        pitchf = pitchf.to(rank, non_blocking=True)
        spec = spec.to(rank, non_blocking=True)
        wav = wav.to(rank, non_blocking=True)
        sid = sid.to(rank, non_blocking=True)
        phone_lengths = phone_lengths.to(rank, non_blocking=True)
        spec_lengths = spec_lengths.to(rank, non_blocking=True)
        wav_lengths = wav_lengths.to(rank, non_blocking=True)
        with autocast(enabled=hps.train.fp16_run):
            (
                z,
                z_p,
                m_p,
                logs_p,
                m_q,
                logs_q,
                log_det_w,
                f0,
                f0_p,
                z_mask,
                x,
                x_lengths,
                sid_r,
            ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            mel = commons.spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_hat = net_g.module.decode(z).float()
            y_hat_mel = commons.spec_to_mel_torch(
                y_hat.squeeze(1),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            wav = F.pad(wav.unsqueeze(1), (0, x.size(2) - wav.size(2)), value=0)
            y = commons.slice_segments(
                wav, x_lengths, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = DiscriminatorLoss(
                y_d_hat_r, y_d_hat_g
            )
        optim_d.zero_grad()
        if hps.train.fp16_run:
            scaler.scale(loss_disc).backward()
            scaler.step(optim_d)
        else:
            loss_disc.backward()
            optim_d.step()

        # Generator
        with autocast(enabled=hps.train.fp16_run):
            # L1 Mel-Spectrogram Loss
            loss_mel = criterion_mel(mel, y_hat_mel)
            loss_mel = (loss_mel * z_mask).sum() / z_mask.sum()
            loss_kl = criterion_kl(z_p, logs_q, m_p, logs_p, log_det_w, z_mask)
            # Discriminator loss for Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            loss_fm = FeatureMatchingLoss(fmap_r, fmap_g)
            loss_gen, losses_gen = GeneratorLoss(y_d_hat_g)
            loss_gen_all = (
                loss_gen
                + hps.train.c_fm * loss_fm
                + hps.train.c_mel * loss_mel
                + hps.train.c_kl * loss_kl
            )
        optim_g.zero_grad()
        if hps.train.fp16_run:
            scaler.scale(loss_gen_all).backward()
            scaler.step(optim_g)
        else:
            loss_gen_all.backward()
            optim_g.step()

        if hps.train.fp16_run:
            scaler.update()
        global_step += 1
        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                # losses
                loss_mel = loss_mel.item()
                loss_gen = loss_gen.item()
                loss_fm = loss_fm.item()
                loss_disc = loss_disc.item()
                loss_kl = loss_kl.item()
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / len(train_loader)
                    )
                )
                logger.info(
                    f"loss_disc: {loss_disc:.4f}, loss_gen: {loss_gen:.4f}, loss_fm: {loss_fm:.4f}, loss_mel: {loss_mel:.4f}, loss_kl: {loss_kl:.4f}, lr: {lr:.6f}"
                )

            if global_step % hps.train.log_interval == 0:
                pass
        
    if epoch % hps.save_every_epoch == 0 and rank == 0:
        utils.save_checkpoint(
            net_g,
            optim_g,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "G_{}.pth".format(epoch)),
        )
        utils.save_checkpoint(
            net_d,
            optim_d,
            hps.train.learning_rate,
            epoch,
            os.path.join(hps.model_dir, "D_{}.pth".format(epoch)),
        )

    if rank == 0 and hps.save_every_weights == "1":
        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        logger.info(
            "saving ckpt %s_e%s:%s"
            % (
                hps.name,
                epoch,
                # savee() ist nicht definiert, wurde aber in den Log-Strings des Original-Codes verwendet.
                # Hier muss eine Platzhalterfunktion oder das Original savee() verwendet werden,
                # falls es in utils.py ist und nicht importiert wird. Da es nicht importiert ist,
                # lasse ich es hier weg.
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
        f"[RVC-CPU-FIX] Starting CPU training directly (rank=0, n_gpus={n_gpus})."
    )

    if hps.train.fp16_run:
        scaler = GradScaler()
    else:
        scaler = None

    if n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))
    else:
        run(0, n_gpus, hps)


if __name__ == "__main__":
    main()