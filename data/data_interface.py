# _*_ coding: utf-8 _*_

"""
    @Time : 2024/8/1 9:49 
    @Author : smile 笑
    @File : data_interface.py
    @desc :
"""


from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader


class DataInterfaceModule(LightningDataModule):
    def __init__(self, dataset, args):
        super(DataInterfaceModule, self).__init__()

        self.args = args
        self.dataset = dataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.args.select_data == "slake":
                self.train_dataset = self.dataset(self.args, self.args.slake_train_dataset_path, self.args.slake_train_wav_path, "train")
                self.val_dataset = self.dataset(self.args, self.args.slake_test_dataset_path, self.args.slake_test_wav_path, "val")
            if self.args.select_data == "rad":
                self.train_dataset = self.dataset(self.args, self.args.rad_train_dataset_path, self.args.rad_train_wav_path, "train")
                self.val_dataset = self.dataset(self.args, self.args.rad_test_dataset_path, self.args.rad_test_wav_path, "val")
            if self.args.select_data == "ovqa":
                self.train_dataset = self.dataset(self.args, self.args.ovqa_train_dataset_path, self.args.ovqa_train_wav_path, "train")
                self.val_dataset = self.dataset(self.args, self.args.ovqa_test_dataset_path, self.args.ovqa_test_wav_path, "val")
            if self.args.select_data == "path_vqa":
                self.train_dataset = self.dataset(self.args, self.args.path_train_dataset_json_path, self.args.path_train_img_folder_path, self.args.path_vqa_train_wav_path, "train")
                self.val_dataset = self.dataset(self.args, self.args.path_test_dataset_json_path, self.args.path_test_img_folder_path, self.args.path_vqa_test_wav_path, "val")

        elif stage == "test" or stage is None:
            if self.args.select_data == "slake":
                self.test_dataset = self.dataset(self.args, self.args.slake_test_dataset_path, self.args.slake_test_wav_path, "test")
            if self.args.select_data == "rad":
                self.test_dataset = self.dataset(self.args, self.args.rad_test_dataset_path, self.args.rad_test_wav_path, "test")
            if self.args.select_data == "ovqa":
                self.test_dataset = self.dataset(self.args, self.args.ovqa_test_dataset_path, self.args.ovqa_test_wav_path, "test")
            if self.args.select_data == "path_vqa":
                self.test_dataset = self.dataset(self.args, self.args.path_test_dataset_json_path, self.args.path_test_img_folder_path, self.args.path_vqa_test_wav_path, "test")

    def collate_fn(self, batch):
        image, question, speech_array, ans_id, ans_type_id = list(zip(*batch))
        image = torch.stack(image)
        speech_array = torch.stack(speech_array).squeeze(1)
        ans_id = torch.tensor(ans_id, dtype=torch.int64)
        ans_type_id = torch.tensor(ans_type_id, dtype=torch.int64)

        return image, question, speech_array, ans_id, ans_type_id

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=self.collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, collate_fn=self.collate_fn)

