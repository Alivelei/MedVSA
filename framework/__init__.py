# _*_ coding: utf-8 _*_

"""
    @Time : 2024/8/1 10:09 
    @Author : smile 笑
    @File : __init__.py
    @desc :
"""


from .model_interface import ModelInterfaceModule
from .model_distillation6_interface import ModelDistillation6InterfaceModule


def get_model_module(model_name):
    if model_name == "pmc_clip_speech_base":
        from .model.pmc_clip_distill_former.pmc_clip import pmc_clip_speech_base
        return pmc_clip_speech_base

    if model_name == "pmc_clip_speech_distillation_base":
        from .model.pmc_clip_distill_former.unite_model import pmc_clip_speech_distillation_base
        return pmc_clip_speech_distillation_base

    if model_name == "whisper_resnet_ban_base":
        from .model.resnet_ban.model import whisper_resnet_ban_base
        return whisper_resnet_ban_base

    if model_name == "whisper_resnet_san_base":
        from .model.resnet_san.model import whisper_resnet_san_base
        return whisper_resnet_san_base

    if model_name == "whisper_resnet_mfb_base":
        from .model.resnet_mfb.model import whisper_resnet_mfb_base
        return whisper_resnet_mfb_base


