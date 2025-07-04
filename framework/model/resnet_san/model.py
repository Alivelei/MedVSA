# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/28 16:37
    @Author : smile 笑
    @File : standard_model.py
    @desc : framework.model.resnet_san.
"""


import torch
from torch import nn
from torchvision.models import resnet50
from transformers import WhisperForConditionalGeneration


WHISPER_MODEL_PATH = "./save/whisper-base"


class SanAttention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(SanAttention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = torch.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = torch.softmax(ha, dim=-1)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class MultiModelFusion(nn.Module):
    def __init__(self, san_num_layers=3, n_hidden=1024, san_k=1024):
        super(MultiModelFusion, self).__init__()

        self.san_attention = nn.ModuleList([SanAttention(n_hidden, san_k) for _ in range(san_num_layers)])

    def forward(self, img_res, qus_res):
        vi = img_res
        u = qus_res[:, -1, :]  # 取最后一个节点的输出
        for att_layer in self.san_attention:
            u = att_layer(vi, u)
        return u


class ResNetSanModel(nn.Module):
    def __init__(self, ans_word_size, lstm_n_hidden=1024,
                 dropout=.0, san_num_layers=3, san_k=2048, linear_hid=256):
        super(ResNetSanModel, self).__init__()
        res_model = resnet50(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(res_model.children())[:7])  # 只取resnet的前七层，不需要后面的全连接层

        self.whisper = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH)
        self.context_length = 20
        self.proj_out = nn.Linear(512, 1024)

        self.san_fusion = MultiModelFusion(san_num_layers, lstm_n_hidden, san_k)

        self.ans_model = nn.Sequential(
            nn.Linear(lstm_n_hidden, linear_hid),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_hid, ans_word_size)
        )

    def whisper_forward(self, input_features=None):
        b = input_features.shape[0]
        decoder_input_ids = torch.ones(b, self.context_length, dtype=torch.long).to(input_features.device) * self.whisper.config.decoder_start_token_id
        last_hidden_states = self.whisper.model(input_features, decoder_input_ids=decoder_input_ids)[0]

        speech_features = self.proj_out(last_hidden_states)  # [2, 20, 1028]

        return speech_features

    def forward(self, img, text=None, speech_array=None):
        img_features = self.feature_extractor(img)
        img_features = img_features.flatten(2).transpose(1, 2)

        speech_features = self.whisper_forward(speech_array)
        fusion_features = self.san_fusion(img_features, speech_features)

        res = self.ans_model(fusion_features)

        return res


def whisper_resnet_san_base(**kwargs):
    model = ResNetSanModel(
        lstm_n_hidden=1024, dropout=.2, san_num_layers=3, san_k=2048, linear_hid=256, ans_word_size=kwargs["ans_size"],
    )
    return model


if __name__ == '__main__':
    import time

    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.rand([2, 80, 3000]).cuda()
    model = ResNetSanModel(969).cuda()
    # torch.save(model.state_dict(), "1.pth")

    t1 = time.time()
    res = model(a, speech_array=b)
    t2 = time.time()
    print(t2 - t1)  # 4.44823
    print(sum(x.numel() for x in model.parameters()))  # 94775308
    print(res.shape)




