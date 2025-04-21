# _*_ coding: utf-8 _*_

"""
    @Time : 2024/8/1 10:08 
    @Author : smile 笑
    @File : train.py
    @desc :
"""


import argparse
from data import DataInterfaceModule, RadDatasetModule, SlakeDatasetModule, OVQADatasetModule, PathVQADatasetModule
from framework import ModelInterfaceModule, get_model_module
from framework import ModelDistillation6InterfaceModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def mkdir_println(dir_path, println):
    if os.path.exists(dir_path):
        print(println + "文件夹已创建.")
    else:
        os.mkdir(dir_path)
        print(println + "文件夹创建成功.")


def create_model_module(args):
    model_name = args.model_select + "_" + args.model_size
    model_func = get_model_module(model_name)

    if args.select_data == "slake" or args.select_data == "rad" or args.select_data == "path_vqa" or args.select_data == "ovqa":
        if args.model_distillation == "multi_distillation_loss":
            model = ModelDistillation6InterfaceModule(model=model_func, args=args)
        else:
            model = ModelInterfaceModule(model=model_func, args=args)

    args.default_root_dir = os.path.join(args.default_root_dir, model_name + "/")
    mkdir_println(args.default_root_dir, model_name + "根")

    args.train_epoch_effect_path = os.path.join(args.default_root_dir, args.train_epoch_effect_path)
    args.test_epoch_effct_path = os.path.join(args.default_root_dir, args.test_epoch_effect_path)
    mkdir_println(args.train_epoch_effect_path, model_name + "_param")

    args.best_model_path = os.path.join(args.default_root_dir, args.best_model_path)
    mkdir_println(args.best_model_path, model_name + "_train_best_model")

    args.test_best_model_path = os.path.join(args.default_root_dir, args.test_best_model_path)
    mkdir_println(args.test_best_model_path, model_name + "_test_best_model")

    return model, args


def dataset_select(args):
    if args.select_data == "slake":
        db = DataInterfaceModule(SlakeDatasetModule, args)
    elif args.select_data == "rad":
        db = DataInterfaceModule(RadDatasetModule, args)
    elif args.select_data == "ovqa":
        db = DataInterfaceModule(OVQADatasetModule, args)
    elif args.select_data == "path_vqa":
        db = DataInterfaceModule(PathVQADatasetModule, args)

    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        version=args.model_select + "_" + args.model_size + "_" + args.select_data + "_" + str(args.version),
        name="train_logs"
    )

    return db, logger


def main(args):
    seed_everything(args.random_seed, True)

    model, args = create_model_module(args)
    db, logger = dataset_select(args)

    train_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=0,
        dirpath=os.path.join(args.best_model_path, str(logger.version)),
        filename="{epoch}-{train_loss:.4f}",
        save_last=True,
    )

    test_checkpoint_callback = ModelCheckpoint(
        monitor="test_total_acc",
        mode="max",
        save_top_k=1,
        dirpath=os.path.join(args.test_best_model_path, str(logger.version)),
        filename="{epoch}-{test_total_acc:.4f}",
        save_weights_only=True,
    )

    epoch_effect_path = os.path.join(args.train_epoch_effect_path, str(logger.version))
    mkdir_println(epoch_effect_path, "model_param_version")
    args.train_epoch_effect_path = os.path.join(epoch_effect_path, "train_epoch_effect.json")
    args.test_epoch_effect_path = os.path.join(epoch_effect_path, "test_epoch_effect.json")

    trainer = Trainer(
        gpus=args.device_ids,
        max_epochs=args.epochs,
        strategy="ddp",
        enable_checkpointing=True,
        check_val_every_n_epoch=5,
        logger=logger,
        sync_batchnorm=True,
        callbacks=[train_checkpoint_callback, test_checkpoint_callback],
        gradient_clip_val=0.5 if args.model_distillation is None else None,  # 如果使用手动梯度，则不能在trainer中定义梯度裁剪
        resume_from_checkpoint=args.resume_from_checkpoint if os.path.exists(args.resume_from_checkpoint) else None,
    )

    trainer.fit(model, db)

    return


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_select", default="pmc_clip_speech_distillation", choices=[
        "pmc_clip_speech_distillation", "whisper_resnet_mfb", "whisper_resnet_san", "whisper_resnet_ban",
        "pmc_clip_speech"
    ])
    parser.add_argument("--model_distillation", default="multi_distillation_loss", choices=[
        "multi_distillation_loss", None
    ])
    parser.add_argument("--model_size", default="base", choices=["base", "large", "huge"])
    parser.add_argument("--select_data", default="path_vqa", choices=["slake", "rad", "path_vqa", "ovqa"])
    parser.add_argument("--load_pre_model", default=True, choices=[True, False])
    parser.add_argument("--freeze_model", default=False, choices=[True, False])
    parser.add_argument("--select_mix", default="img_cutmix", choices=["img_mixup", "img_cutmix"])
    parser.add_argument("--mix_probability", default=0)
    parser.add_argument("--mix_alpha_1", default=5, type=int)
    parser.add_argument("--mix_alpha_2", default=1, type=int)
    parser.add_argument("--version", default="no_mix_no_aug_m6")

    # configure
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device_ids", default=[2, 3, 4])
    parser.add_argument("--num_workers", default=4, type=int)

    # model
    parser.add_argument("--training_temp", default=1, choices=[1, 3, 7])
    parser.add_argument("--training_alpha", default=0.3, choices=[0.3, 0.5, 0.7])
    parser.add_argument("--tch_learning_rate", default=0.00001, type=float)
    parser.add_argument("--tch_weights_decay", default=0.05, type=float)
    parser.add_argument("--std_learning_rate", default=0.00001, type=float)
    parser.add_argument("--std_weights_decay", default=0.05, type=float)
    parser.add_argument("--random_seed", default=1024, type=int)

    # constant_image
    parser.add_argument("--img_rotation", default=15, type=int)
    parser.add_argument("--resized_crop_left", default=0.6, type=float)
    parser.add_argument("--resized_crop_right", default=1.0, type=float)
    parser.add_argument("--blur", default=[0.1, 2.0])
    parser.add_argument("--b_size", default=[5, 5])
    parser.add_argument("--blur_p", default=0.5, type=float)
    parser.add_argument("--apply_p", default=0.8, type=float)
    parser.add_argument("--img_flip", default=0.5, type=float)
    parser.add_argument("--brightness", default=0.4, type=float)
    parser.add_argument("--contrast", default=0.4, type=float)
    parser.add_argument("--saturation", default=0.4, type=float)
    parser.add_argument("--hue", default=0.4, type=float)
    parser.add_argument("--grayscale", default=0.2, type=float)

    # rand aug image
    parser.add_argument("--general_rand_aug", default=False)
    parser.add_argument("--resized_crop_scale_left", default=0.6, type=float)
    parser.add_argument("--resized_crop_scale_right", default=1, type=float)
    parser.add_argument("--ra_n", default=2)
    parser.add_argument("--ra_m", default=12)
    parser.add_argument("--img_jitter", default=0.2, type=float)
    parser.add_argument("--reprob", default=0.2)

    # configure
    parser.add_argument("--epochs", default=15000, type=int)
    parser.add_argument("--qus_seq_len", default=20, type=int)
    parser.add_argument("--answer_open", default=0, type=int)
    parser.add_argument("--answer_close", default=1, type=int)
    parser.add_argument("--train_epoch_effect_path", default="param")
    parser.add_argument("--test_epoch_effect_path", default="param")

    parser.add_argument("--best_model_path", default="best_model")
    parser.add_argument("--test_best_model_path", default="test_best_model")
    parser.add_argument("--default_root_dir", default="./save/")
    parser.add_argument("--resume_from_checkpoint", default="./save/model/best_model/last.ckpt")

    # slake dataset
    parser.add_argument("--slake_train_wav_path", default="./data/ref/Slake1.0/wavs/")
    parser.add_argument("--slake_test_wav_path", default="./data/ref/Slake1.0/test_wavs/")
    parser.add_argument("--slake_ans_ws_path", default="./save/ws/slake_ans_ws.pkl")
    parser.add_argument("--slake_ans_word_size", default=222, type=int)
    parser.add_argument("--slake_train_dataset_path", default="./data/ref/Slake1.0/train.json")
    parser.add_argument("--slake_test_dataset_path", default="./data/ref/Slake1.0/test.json")
    parser.add_argument("--slake_dataset_xm_path", default="./data/ref/Slake1.0/imgs/xmlab")

    # rad dataset
    parser.add_argument("--rad_train_wav_path", default="./data/ref/rad/wavs/")
    parser.add_argument("--rad_test_wav_path", default="./data/ref/rad/test_wavs/")
    parser.add_argument("--rad_ans_word_size", default=475, type=int)
    parser.add_argument("--rad_ans_ws_path", default="./save/ws/rad_ans_ws.pkl")
    parser.add_argument("--rad_images_path", default="./data/ref/rad/images")
    parser.add_argument("--rad_train_dataset_path", default="./data/ref/rad/trainset.json")
    parser.add_argument("--rad_test_dataset_path", default="./data/ref/rad/testset.json")

    # path_vqa dataset
    parser.add_argument("--path_vqa_train_wav_path", default="./data/ref/PathVQA/qas/train/wavs/")
    parser.add_argument("--path_vqa_test_wav_path", default="./data/ref/PathVQA/qas/test/wavs/")
    parser.add_argument("--path_vqa_ans_word_size", default=4092, type=int)
    parser.add_argument("--path_vqa_ans_ws_path", default="./save/ws/path_vqa_ans_ws.pkl")
    parser.add_argument("--path_train_img_folder_path", default="./data/ref/PathVQA/images/train")
    parser.add_argument("--path_test_img_folder_path", default="./data/ref/PathVQA/images/test")
    parser.add_argument("--path_train_dataset_json_path", default="./data/ref/PathVQA/qas/train/train.json")
    parser.add_argument("--path_test_dataset_json_path", default="./data/ref/PathVQA/qas/test/test.json")

    # ovqa dataset
    parser.add_argument("--ovqa_train_wav_path", default="./data/ref/OVQA_publish/wavs/")
    parser.add_argument("--ovqa_test_wav_path", default="./data/ref/OVQA_publish/test_wavs/")
    parser.add_argument("--ovqa_ans_word_size", default=707, type=int)
    parser.add_argument("--ovqa_ans_ws_path", default="./save/ws/ovqa_ans_ws.pkl")
    parser.add_argument("--ovqa_images_path", default="./data/ref/OVQA_publish/img")
    parser.add_argument("--ovqa_train_dataset_path", default="./data/ref/OVQA_publish/trainset.json")
    parser.add_argument("--ovqa_test_dataset_path", default="./data/ref/OVQA_publish/testset.json")

    # image
    parser.add_argument("--img_height", default=224, type=int)
    parser.add_argument("--img_width", default=224, type=int)
    parser.add_argument("--slake_img_mean", default=[0.38026, 0.38026, 0.38026])
    parser.add_argument("--slake_img_std", default=[0.2979, 0.2979, 0.2979])
    parser.add_argument("--rad_img_mean", default=[0.33640, 0.33630, 0.33610])
    parser.add_argument("--rad_img_std", default=[0.29664, 0.29659, 0.29642])
    parser.add_argument("--path_vqa_img_mean", default=[0.6755, 0.5576, 0.6504])
    parser.add_argument("--path_vqa_img_std", default=[0.3275, 0.3081, 0.3212])
    parser.add_argument("--ovqa_img_mean", default=[0.2016, 0.1895, 0.1793])
    parser.add_argument("--ovqa_img_std", default=[0.3169, 0.3032, 0.2927])

    args = parser.parse_args()

    main(args)

    # dl = DataInterfaceModule(PathVQADatasetModule, args)
    # dl.setup(stage="test")
    #
    # for idx, (img, qus, speech, ans, ans_flag) in enumerate(dl.test_dataloader()):
    #     print(img.shape)
    #     print(qus)
    #     print(speech.shape)  # torch.Size([128, 1, 80, 3000])
    #     print(ans.shape)
    #     print(ans_flag.shape)
    #
    # dl.teardown(stage="test")
