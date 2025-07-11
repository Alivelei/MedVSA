# _*_ coding: utf-8 _*_

"""
    @Time : 2024/3/4 14:11 
    @Author : smile 笑
    @File : unite_model.py
    @desc :
"""


from .model import create_pmc_clip_model
import torch
from torch import nn
from transformers import WhisperForConditionalGeneration
import torch.nn.functional as F


PMC_CLIP_MODEL_PATH = "./save/pmc_clip_ckpt/checkpoint.pt"
WHISPER_MODEL_PATH = "./save/whisper-base"


class PMCCLIPModel(nn.Module):
    def __init__(self, load_pre_model=True, freeze_pre_model=False, ans_word_size=223):
        super(PMCCLIPModel, self).__init__()

        self.teacher_model, args = create_pmc_clip_model()

        if load_pre_model:
            args.resume = PMC_CLIP_MODEL_PATH
            args.distributed = False
            checkpoint = torch.load(args.resume, map_location="cpu")  # args.device
            sd = checkpoint["state_dict"]

            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            self.teacher_model.load_state_dict(sd, strict=False)  # 成功加载，牛逼

        if freeze_pre_model:
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        self.teacher_res_linear = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, ans_word_size)
        )

    def forward(self, img, qus, speech_array=None):
        batch = dict()

        batch["images"] = img
        batch["bert_input"] = qus
        batch["bert_label"] = qus

        clip_content = self.teacher_model(batch)
        image_features, text_features, fusion_features = clip_content["image_features"], clip_content["text_features"], clip_content["bert_prediction"]

        # fusion_features = F.normalize(fusion_features, dim=-1)
        teacher_prediction = self.teacher_res_linear(fusion_features.mean(1))  # [2, 223]

        # [2, 223]; [2, 768], [2, 768], [2, 768]
        return teacher_prediction, [image_features, text_features, fusion_features.mean(1)]


class PMCCLIPStudentModel(nn.Module):
    def __init__(self, ans_word_size=223):
        super(PMCCLIPStudentModel, self).__init__()

        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH)
        self.context_length = 20
        self.proj_out = nn.Linear(512, 768)

        stu_model = PMCCLIPModel()
        self.encode_image = stu_model.teacher_model.visual
        self.fusion_module = stu_model.teacher_model.fusion_module

        self.cls_id = 2
        self.img_special_token = nn.Parameter(torch.zeros(1, 1, 768))

        self.student_res_linear = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, ans_word_size)
        )

    def whisper_forward(self, input_features=None):
        b = input_features.shape[0]
        decoder_input_ids = torch.ones(b, self.context_length, dtype=torch.long).to(input_features.device) * self.whisper.config.decoder_start_token_id
        last_hidden_states = self.whisper.model(input_features, decoder_input_ids=decoder_input_ids)[0]

        speech_features = self.proj_out(last_hidden_states)  # [2, 77, 768]

        return speech_features

    def forward(self, image, text=None, speech_array=None):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features['image_features'], dim=-1)

        speech_features = self.whisper_forward(speech_array)
        text_features = F.normalize(speech_features, dim=-1)  # [2, 77, 768]

        # Fusion Module
        img = torch.unsqueeze(image_features, 1)  # [128, 1 ,768]
        B, _len, _dim = text_features.shape
        img_special_tokens = self.img_special_token.expand(B, -1, -1)  # [128, 1, embed_dim]
        x = torch.cat([text_features, img_special_tokens, img], dim=1)  # [128, 77+1+1, 768]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.fusion_module(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, :-2, :]  # Remove token [img_special_token, img]

        # fusion_features = F.normalize(x, dim=-1)
        fusion_features = x
        teacher_prediction = self.student_res_linear(fusion_features.mean(1))

        return teacher_prediction, [image_features, text_features.mean(1), fusion_features.mean(1)]


def pmc_clip_speech_distillation_base(**kwargs):
    teacher_model = PMCCLIPModel(load_pre_model=kwargs["load_pre_model"], freeze_pre_model=kwargs["freeze_model"],
                                 ans_word_size=kwargs["ans_size"])
    student_model = PMCCLIPStudentModel(ans_word_size=kwargs["ans_size"])

    return [teacher_model, student_model]


if __name__ == '__main__':
    from transformers.pipelines.audio_utils import ffmpeg_read
    from transformers import WhisperProcessor
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    images = torch.ones([2, 3, 224, 224]).cuda()
    bert_input = [
        "I've been waiting for a this course my whole life.",
        "HOW TO TREAT THE DISEASE LOCATED IN THE CENTER OF THIS IMAGE".lower()
    ]

    batch = dict()

    batch["images"] = images
    batch["bert_input"] = bert_input
    batch["bert_label"] = bert_input
    # teacher_model = PMCCLIPModel().cuda()
    # print(sum(x.numel() for x in teacher_model.parameters()))  # 199891264

    inputs = "../../../11.wav"

    with open(inputs, "rb") as f:
        inputs = f.read()

    ffmpeg = ffmpeg_read(inputs, 16000)
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
    input_features = processor([ffmpeg, ffmpeg], sampling_rate=16000, return_tensors="pt").input_features.cuda()

    model = PMCCLIPStudentModel().cuda()
    prediction, [image_features, text_features, fusion_features] = model(images, 0, input_features)
    print(prediction.shape, image_features.shape, text_features.shape, fusion_features.shape)
    print(sum(x.numel() for x in model.parameters()))  # 139305535   图像模型+语音模型+融合模型

