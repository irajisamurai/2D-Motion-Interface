from src.data.HumanML3D import MotionDatasetVQ_2D
from src.data.humanml.dataset_t2m_eval import Text2MotionDatasetEval_2D
from os.path import join as pjoin
import numpy as np
from src.data.utils import humanml3d_collate_2d
from src.data.humanml.utils.word_vectorizer import WordVectorizer
from torch.utils.data import DataLoader



def build_humanml3d_dataloader_2d(cfg):
        data_root = cfg.DATASET.HUMANML3D.ROOT
        dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        mean = np.load(pjoin(dis_data_root, "mean.npy"))
        std = np.load(pjoin(dis_data_root, "std.npy"))
        mean_2d = np.load(pjoin(dis_data_root, "mean_2d.npy"))
        std_2d = np.load(pjoin(dis_data_root, "std_2d.npy"))

        _train_dataset = MotionDatasetVQ_2D(
                        data_root=data_root,
                        split='train',
                        mean=mean,
                        std=std,
                        mean_2d=mean_2d,
                        std_2d=std_2d,
                        max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
                        min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
                        win_size=64,
                        unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,)
        w_vectorizer = WordVectorizer(
                cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        _val_dataset = Text2MotionDatasetEval_2D(
                data_root=data_root,
                split='test',
                mean=mean,
                std=std,
                mean_2d=mean_2d,
                std_2d=std_2d,
                max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
                min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
                win_size=64,
                unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
                w_vectorizer=w_vectorizer,)

        _train_dataloader = DataLoader(
                dataset=_train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                collate_fn=humanml3d_collate_2d,
                persistent_workers=True
        )

        _val_dataloader = DataLoader(
                dataset=_val_dataset,
                batch_size=cfg.EVAL.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                collate_fn=humanml3d_collate_2d,
                persistent_workers=True
        )
        return _train_dataloader, _val_dataloader

def build_humanml3d_motion_x_dataloader_2d(cfg):
        data_root = cfg.DATASET.HUMANML3D.ROOT
        test_data_root = cfg.DATASET.MotionX.ROOT
        dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        mean = np.load(pjoin(dis_data_root, "mean.npy"))
        std = np.load(pjoin(dis_data_root, "std.npy"))
        mean_2d = np.load(pjoin(dis_data_root, "mean_2d.npy"))
        std_2d = np.load(pjoin(dis_data_root, "std_2d.npy"))
        add_noise = cfg.DATASET.MotionX.NOISE
        add_stretch = cfg.DATASET.MotionX.STRETCH
        sigma = cfg.DATASET.MotionX.params.sigma
        p_noise = cfg.DATASET.MotionX.params.p_noise
        p_stretch = cfg.DATASET.MotionX.params.p_stretch
        stretch_range = cfg.DATASET.MotionX.params.stretch_range
        alpha = cfg.DATASET.MotionX.params.alpha

        _train_dataset = MotionDatasetVQ_2D(
                        data_root=data_root,
                        split='train',
                        mean=mean,
                        std=std,
                        mean_2d=mean_2d,
                        std_2d=std_2d,
                        max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
                        min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
                        win_size=64,
                        unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
                        add_noise=add_noise,
                        add_stretch=add_stretch,
                        sigma=sigma,
                        p_noise=p_noise,
                        p_stretch=p_stretch,
                        stretch_range=stretch_range,
                        alpha=alpha)
        w_vectorizer = WordVectorizer(
                cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")
        _val_dataset = Text2MotionDatasetEval_2D(
                data_root=test_data_root,
                split='test',
                mean=mean,
                std=std,
                mean_2d=mean_2d,
                std_2d=std_2d,
                max_motion_length=cfg.DATASET.HUMANML3D.MAX_MOTION_LEN,
                min_motion_length=cfg.DATASET.HUMANML3D.MIN_MOTION_LEN,
                win_size=64,
                unit_length=cfg.DATASET.HUMANML3D.UNIT_LEN,
                w_vectorizer=w_vectorizer,)

        _train_dataloader = DataLoader(
                dataset=_train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                collate_fn=humanml3d_collate_2d,
                persistent_workers=True
        )

        _val_dataloader = DataLoader(
                dataset=_val_dataset,
                batch_size=cfg.EVAL.BATCH_SIZE,
                shuffle=False,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                collate_fn=humanml3d_collate_2d,
                persistent_workers=True
        )
        return _train_dataloader, _val_dataloader