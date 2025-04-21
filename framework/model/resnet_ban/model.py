# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/28 16:37
    @Author : smile 笑
    @File : standard_model.py
    @desc : framework.model.resnet_ban.
"""


import torch
from torch import nn
from torchvision.models import resnet50
from .ban import BiAttention, BanBiResNet
from transformers import WhisperForConditionalGeneration


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


class BanMultiModelFusion(nn.Module):
    def __init__(self, x_dim=256, y_dim=1024, z_dim=1024, glimpse=4, v_dim=256, hid_dim=1024):
        super(BanMultiModelFusion, self).__init__()

        self.bi_attn = BiAttention(x_dim, y_dim, z_dim, glimpse)
        self.ban_bi = BanBiResNet(v_dim, hid_dim, glimpse)

    def forward(self, img_res, qus_res):
        p, logits = self.bi_attn(img_res, qus_res)
        res = self.ban_bi(img_res, qus_res, p)

        return res


class ResNetBanModel(nn.Module):
    def __init__(self, ans_word_size, out_channel=256, dropout=.0, x_dim=256, y_dim=1024, z_dim=1024,
                 glimpse=16, v_dim=256, hid_dim=1024):
        super(ResNetBanModel, self).__init__()
        res_model = resnet50(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(res_model.children())[:7])  # 只取resnet的前七层，不需要后面的全连接层
        self.down_pooling = nn.Sequential(
            CBR(1024, out_channel, stride=1),  # resnet的输出结果[b, 1024, 14, 14]
        )

        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH)
        self.context_length = 20
        self.proj_out = nn.Linear(512, 1024)

        self.ban_fusion = BanMultiModelFusion(x_dim, y_dim, z_dim, glimpse, v_dim, hid_dim)

        self.ans_model = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hid_dim, ans_word_size)
        )

    def whisper_forward(self, input_features=None):
        b = input_features.shape[0]
        decoder_input_ids = torch.ones(b, self.context_length, dtype=torch.long).to(input_features.device) * self.whisper.config.decoder_start_token_id
        last_hidden_states = self.whisper.model(input_features, decoder_input_ids=decoder_input_ids)[0]

        speech_features = self.proj_out(last_hidden_states)  # [2, 20, 1028]

        return speech_features

    def forward(self, img, qus_embed=None, speech_array=None):
        img_features = self.down_pooling(self.feature_extractor(img))
        img_features = img_features.flatten(2).transpose(1, 2).contiguous()

        speech_features = self.whisper_forward(speech_array)
        fusion_features = self.ban_fusion(img_features, speech_features)

        res = self.ans_model(fusion_features)

        return res


def whisper_resnet_ban_base(**kwargs):
    model = ResNetBanModel(
        out_channel=256, dropout=.3, x_dim=256, y_dim=1024, z_dim=1024, glimpse=4, v_dim=256, hid_dim=1024,
        ans_word_size=kwargs["ans_size"],
    )
    return model


if __name__ == '__main__':
    import time

    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.rand([2, 80, 3000]).cuda()
    model = ResNetBanModel(223).cuda()
    t1 = time.time()
    res = model(a, speech_array=b)
    t2 = time.time()
    print(t2 - t1)  # 2.9607625007629395
    print(sum(x.numel() for x in model.parameters()))  # 127282801


