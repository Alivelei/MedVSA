# _*_ coding: utf-8 _*_

"""
    @Time : 2024/8/1 9:50 
    @Author : smile 笑
    @File : datasets.py
    @desc :
"""


from torch.utils.data import Dataset
from transformers import WhisperProcessor
from transformers.pipelines.audio_utils import ffmpeg_read
import json
import pickle
import os
import torchvision.transforms as tfs
from PIL import Image


def sentence_to_word(sentence, qus=True):
    if qus:
        if "▁" in sentence:
            queries = str(sentence).upper().split("▁")
        else:
            queries = str(sentence).upper().split(" ")  # 将问题进行切分，且都转换为大写
    else:
        queries = str(sentence).lower().strip(" ").strip(".")  # 将答案都转换为小写
    return queries


def word_id_transform(word_id_dict, sentence, max_len=None, pad_tag="PAD", unk=1):

    if max_len is not None:
        if max_len > len(sentence):
            sentence = sentence + [pad_tag] * (max_len - len(sentence))
        if max_len < len(sentence):
            sentence = sentence[:max_len]

    return [word_id_dict.get(word, unk) for word in sentence]


def train_aug_img(img, args, img_mean, img_std):
    if args.general_rand_aug:
        aug = tfs.Compose([
            tfs.RandomResizedCrop(args.img_height, scale=(args.resized_crop_scale_left, args.resized_crop_scale_right),
                                  ratio=(1.0, 1.0)),
            tfs.RandomHorizontalFlip(p=args.img_flip),
            tfs.RandAugment(args.ra_n, args.ra_m),
            tfs.ColorJitter(args.img_jitter, args.img_jitter, args.img_jitter),
            tfs.ToTensor(),
            tfs.Normalize(img_mean, img_std),
            tfs.RandomErasing(args.reprob)
        ])
    else:
        aug = tfs.Compose([
            tfs.RandomResizedCrop(args.img_height, scale=(args.resized_crop_left, args.resized_crop_right)),
            tfs.RandomApply([tfs.GaussianBlur(kernel_size=args.b_size, sigma=args.blur)], p=args.blur_p),
            tfs.RandomGrayscale(args.grayscale),
            tfs.RandomApply([
                tfs.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue)
            ], p=args.apply_p),
            tfs.RandomRotation(args.img_rotation),
            tfs.RandomHorizontalFlip(args.img_flip),
            tfs.ToTensor(),
            tfs.Normalize(img_mean, img_std)
        ])

    return aug(img)


def test_aug_img(img, args, img_mean, img_std):
    aug = tfs.Compose([
        tfs.Resize([args.img_height, args.img_width]),
        tfs.ToTensor(),
        tfs.Normalize(img_mean, img_std)
    ])

    return aug(img)


class SlakeDatasetModule(Dataset):
    def __init__(self, args, dataset_path, wavs_path, mode):
        self.args = args

        self.mode = mode
        self.xm_path = args.slake_dataset_xm_path
        self.queries = json.load(open(dataset_path, encoding="utf-8"))
        self.queries = [query for query in self.queries if query["q_lang"] == "en"]

        self.wavs_path = wavs_path
        self.processor = WhisperProcessor.from_pretrained("./save/whisper-base")

        self.ans_ws = pickle.load(open(args.slake_ans_ws_path, "rb"))

        self.slake_img_mean = args.slake_img_mean
        self.slake_img_std = args.slake_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]
        qid = query["qid"]

        img_path = os.path.join(self.xm_path + str(query["img_id"]), "source.jpg")
        wav_path = os.path.join(self.wavs_path, str(qid) + ".wav")

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.slake_img_mean, self.slake_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.slake_img_mean, self.slake_img_std)

        with open(wav_path, "rb") as f:
            inputs = f.read()
        speech_array = ffmpeg_read(inputs, 16000)
        speech_features = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features

        question = sentence_to_word(query["question"], True)

        answer = sentence_to_word(query["answer"], False)
        answer_type = query["answer_type"]

        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, " ".join(question), speech_features, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)


class RadDatasetModule(Dataset):
    def __init__(self, args, rad_dataset_path, wavs_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.rad_images_path
        self.queries = json.load(open(rad_dataset_path, encoding="utf-8"))
        self.wavs_path = wavs_path

        self.processor = WhisperProcessor.from_pretrained("./save/whisper-base")

        self.ans_ws = pickle.load(open(args.rad_ans_ws_path, "rb"))

        self.rad_img_mean = args.rad_img_mean
        self.rad_img_std = args.rad_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]
        qid = query["qid"]

        img_path = os.path.join(self.images_path, str(query["image_name"]))
        wav_path = os.path.join(self.wavs_path, str(qid) + ".wav")

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.rad_img_mean, self.rad_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.rad_img_mean, self.rad_img_std)

        with open(wav_path, "rb") as f:
            inputs = f.read()
        speech_array = ffmpeg_read(inputs, 16000)
        speech_features = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features

        question = sentence_to_word(query["question"], True)

        answer = sentence_to_word(str(query["answer"]), False)
        answer_type = query["answer_type"]

        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, " ".join(question), speech_features, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)


class OVQADatasetModule(Dataset):
    def __init__(self, args, ovqa_dataset_path, wavs_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.ovqa_images_path
        self.queries = json.load(open(ovqa_dataset_path, encoding="utf-8"))
        self.wavs_path = wavs_path

        self.processor = WhisperProcessor.from_pretrained("./save/whisper-base")

        self.ans_ws = pickle.load(open(args.ovqa_ans_ws_path, "rb"))

        self.ovqa_img_mean = args.ovqa_img_mean
        self.ovqa_img_std = args.ovqa_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]
        qid = query["qid"]

        img_path = os.path.join(self.images_path, str(query["image_name"]))
        wav_path = os.path.join(self.wavs_path, str(qid) + ".wav")

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.ovqa_img_mean, self.ovqa_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.ovqa_img_mean, self.ovqa_img_std)

        with open(wav_path, "rb") as f:
            inputs = f.read()
        speech_array = ffmpeg_read(inputs, 16000)
        speech_features = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features

        question = sentence_to_word(query["question"], True)

        answer = sentence_to_word(str(query["answer"]), False)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, " ".join(question), speech_features, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)


class PathVQADatasetModule(Dataset):
    def __init__(self, args, json_dataset_path, img_folder_path, wavs_path, mode):
        self.args = args
        self.mode = mode

        self.queries = json.load(open(json_dataset_path, encoding="utf-8"))
        self.images_path = img_folder_path

        self.wavs_path = wavs_path

        self.processor = WhisperProcessor.from_pretrained("./save/whisper-base")

        self.ans_ws = pickle.load(open(args.path_vqa_ans_ws_path, "rb"))

        self.path_vqa_img_mean = args.path_vqa_img_mean
        self.path_vqa_img_std = args.path_vqa_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]
        qid = query["qid"]

        img_path = os.path.join(self.images_path, str(query["image"]) + ".jpg")
        wav_path = os.path.join(self.wavs_path, str(qid) + ".wav")

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.path_vqa_img_mean, self.path_vqa_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.path_vqa_img_mean, self.path_vqa_img_std)

        with open(wav_path, "rb") as f:
            inputs = f.read()
        speech_array = ffmpeg_read(inputs, 16000)
        speech_features = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").input_features

        question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(str(query["answer"]), False)

        if answer.lower() == "no" or answer.lower() == "yes":
            answer_type_id = self.args.answer_close
        else:
            answer_type_id = self.args.answer_open

        ans_id = word_id_transform(self.ans_ws, [answer], unk=0)

        return image, " ".join(question), speech_features, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)
