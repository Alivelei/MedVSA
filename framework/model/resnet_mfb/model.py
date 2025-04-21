# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/28 16:37
    @Author : smile 笑
    @File : standard_model.py
    @desc : framework.model.resnet_mfb.
"""


import torch
from torch import nn
from torchvision.models import resnet50
from transformers import WhisperForConditionalGeneration
from .mfb import CoAtt
from .config import Cfgs


WHISPER_MODEL_PATH = "./save/whisper-base"


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1):
        super(CBR, self).__init__()
        self.cbl = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.cbl(x)


class ResNetMFBModel(nn.Module):
    def __init__(self, ans_word_size, lstm_n_hidden=1024, dropout=.0, linear_dim=256):
        super(ResNetMFBModel, self).__init__()
        res_model = resnet50(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(res_model.children())[:7])  # 只取resnet的前七层，不需要后面的全连接层

        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH)
        self.context_length = 20
        self.proj_out = nn.Linear(512, 1024)

        self.mfb_fusion = CoAtt(Cfgs())

        self.ans_model = nn.Sequential(
            nn.Linear(lstm_n_hidden, linear_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_dim, ans_word_size)
        )

    def whisper_forward(self, input_features=None):
        b = input_features.shape[0]
        decoder_input_ids = torch.ones(b, self.context_length, dtype=torch.long).to(input_features.device) * self.whisper.config.decoder_start_token_id
        last_hidden_states = self.whisper.model(input_features, decoder_input_ids=decoder_input_ids)[0]

        speech_features = self.proj_out(last_hidden_states)  # [2, 20, 1028]

        return speech_features

    def forward(self, img, qus_embed=None, speech_array=None):
        img_features = self.feature_extractor(img)
        img_features = img_features.flatten(2)

        speech_features = self.whisper_forward(speech_array)
        # print(img_features.shape, qus_features.shape)
        fusion_features = self.mfb_fusion(img_features, speech_features)

        res = self.ans_model(fusion_features)

        return res


def whisper_resnet_mfb_base(**kwargs):
    model = ResNetMFBModel(
        ans_word_size=kwargs["ans_size"],  lstm_n_hidden=1024, dropout=.3, linear_dim=256,
    )
    return model


if __name__ == '__main__':
    import time

    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.rand([2, 80, 3000]).cuda()
    model = ResNetMFBModel(223).cuda()
    # torch.save(model.state_dict(), "1.pth")

    t1 = time.time()
    res = model(a, speech_array=b)
    t2 = time.time()
    print(t2 - t1)  # 2.879370

    print(sum(x.numel() for x in model.parameters()))  # 107036451


