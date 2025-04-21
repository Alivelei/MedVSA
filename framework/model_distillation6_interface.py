# _*_ coding: utf-8 _*_

"""
    @Time : 2024/3/6 12:28 
    @Author : smile 笑
    @File : model_distillation6_interface.py
    @desc :
"""


import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
from framework.metrics.method import compute_batch_score
from framework.metrics.epoch_res import save_epoch_res
import torch.distributed as dist
import torch.nn.functional as F


class ModelDistillation6InterfaceModule(pl.LightningModule):
    def __init__(self, model, args):
        super(ModelDistillation6InterfaceModule, self).__init__()

        # self.hparams = args
        self.save_hyperparameters()

        self.hard_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        self.temp = args.training_temp
        self.alpha = args.training_alpha

        self.automatic_optimization = False

        if args.select_data == "slake":
            ans_word_size = args.slake_ans_word_size
        elif args.select_data == "rad":
            ans_word_size = args.rad_ans_word_size
        elif args.select_data == "path_vqa":
            ans_word_size = args.path_vqa_ans_word_size
        elif args.select_data == "ovqa":
            ans_word_size = args.ovqa_ans_word_size

        self.teacher_model, self.student_model = model(load_pre_model=args.load_pre_model, freeze_model=args.freeze_model, ans_size=ans_word_size)

    def on_train_epoch_start(self):
        # 保存每个epoch的close、open准确值和数量
        self.clip_train_e_close_acc = self.clip_train_e_open_acc = self.clip_train_e_total_acc = 0
        self.clip_train_e_close_nums = self.clip_train_e_open_nums = self.clip_train_e_total_nums = 0
        self.train_e_close_acc = self.train_e_open_acc = self.train_e_total_acc = 0
        self.train_e_close_nums = self.train_e_open_nums = self.train_e_total_nums = 0

    def on_validation_epoch_start(self):
        self.clip_test_e_close_acc = self.clip_test_e_open_acc = self.clip_test_e_total_acc = 0
        self.clip_test_e_close_nums = self.clip_test_e_open_nums = self.clip_test_e_total_nums = 0
        self.test_e_close_acc = self.test_e_open_acc = self.test_e_total_acc = 0
        self.test_e_close_nums = self.test_e_open_nums = self.test_e_total_nums = 0

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def forward(self, img, qus, speech_array):
        return self.student_model(img, qus, speech_array)

    def shares_step_compute_value(self, b_open_acc, b_close_acc, b_total_acc, b_open_nums, b_close_nums, b_total_nums):
        all_b_open_acc = torch.tensor(b_open_acc).cuda()  # sourceTensor.clone().detach()
        all_b_close_acc = torch.tensor(b_close_acc).cuda()
        all_b_total_acc = torch.tensor(b_total_acc).cuda()
        all_b_open_nums = torch.tensor(b_open_nums).cuda()
        all_b_close_nums = torch.tensor(b_close_nums).cuda()
        all_b_total_nums = torch.tensor(b_total_nums).cuda()

        dist.all_reduce(all_b_open_acc)  # 对每个gpu上的value求和
        dist.all_reduce(all_b_close_acc)
        dist.all_reduce(all_b_total_acc)
        dist.all_reduce(all_b_open_nums)
        dist.all_reduce(all_b_close_nums)
        dist.all_reduce(all_b_total_nums)

        b_open_acc = all_b_open_acc / (all_b_open_nums + 1e-10)
        b_close_acc = all_b_close_acc / (all_b_close_nums + 1e-10)  # 加一个足够小的数防止出现0
        b_total_acc = all_b_total_acc / (all_b_total_nums + 1e-10)

        return b_open_acc, b_close_acc, b_total_acc

    def kd_ce_loss(self, logits_S, logits_T, temperature=1):
        '''
        Calculate the cross entropy between logits_S and logits_T
        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        p_T = F.softmax(beta_logits_T, dim=-1)
        loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()

        return loss

    def training_step(self, batch, batch_idx):
        image, qus, speech_array, ans, ans_type = batch

        opt_teacher, opt_student = self.optimizers()

        tea_prediction, tea_content = self.teacher_model(image, qus, speech_array)
        stu_prediction, stu_content = self.student_model(image, qus, speech_array)
        teacher_loss = self.hard_loss(tea_prediction, ans.view(-1))
        student_loss = self.hard_loss(stu_prediction, ans.view(-1))

        tea_img_features, tea_text_features, tea_fus_features = tea_content
        stu_img_features, stu_text_features, stu_fus_features = stu_content

        # 对教师模型的loss进行更新
        opt_teacher.zero_grad()
        self.manual_backward(teacher_loss)
        opt_teacher.step()

        img_features_loss = self.mse_loss(tea_img_features.detach(), stu_img_features)
        text_features_loss = self.mse_loss(tea_text_features.detach(), stu_text_features)
        fus_features_loss = self.mse_loss(tea_fus_features.detach(), stu_fus_features)
        distillation_loss = self.kd_ce_loss(stu_prediction, tea_prediction.detach(), self.temp)

        student_all_loss = (1 - self.alpha) * student_loss + self.alpha * (img_features_loss + text_features_loss + fus_features_loss + distillation_loss)

        # 对学生模型进行更新
        opt_student.zero_grad()
        self.manual_backward(student_all_loss)
        opt_student.step()

        # 对教师模型和学生模型进行手动梯度裁剪
        self.clip_gradients(opt_teacher, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(opt_student, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        _, teacher_pred = tea_prediction.max(-1)  # 取出预测值
        _, student_pred = stu_prediction.max(-1)

        # 计算每个batch的准确率
        open_b_acc, close_b_acc, total_b_acc, open_len, close_len, total_b_len = compute_batch_score(teacher_pred, ans.view(-1), ans_type)

        # 计算open、close、total的每个batch后平均精确率
        self.clip_train_e_open_acc += open_b_acc
        self.clip_train_e_open_nums += open_len
        self.clip_train_e_close_acc += close_b_acc
        self.clip_train_e_close_nums += close_len
        self.clip_train_e_total_acc += total_b_acc
        self.clip_train_e_total_nums += total_b_len

        open_b_acc, close_b_acc, total_b_acc, open_len, close_len, total_b_len = compute_batch_score(student_pred, ans.view(-1), ans_type)
        # 计算open、close、total的每个batch后平均精确率
        self.train_e_open_acc += open_b_acc
        self.train_e_open_nums += open_len
        self.train_e_close_acc += close_b_acc
        self.train_e_close_nums += close_len
        self.train_e_total_acc += total_b_acc
        self.train_e_total_nums += total_b_len

        clip_train_open_acc, clip_train_close_acc, clip_train_total_acc = self.shares_step_compute_value(self.clip_train_e_open_acc, self.clip_train_e_close_acc, self.clip_train_e_total_acc, self.clip_train_e_open_nums, self.clip_train_e_close_nums, self.clip_train_e_total_nums)
        train_open_acc, train_close_acc, train_total_acc = self.shares_step_compute_value(self.train_e_open_acc, self.train_e_close_acc, self.train_e_total_acc, self.train_e_open_nums, self.train_e_close_nums, self.train_e_total_nums)

        self.log("train_teacher_loss_step", teacher_loss, prog_bar=True, on_epoch=False)
        self.log("train_all_student_loss_step", student_all_loss, prog_bar=True, on_epoch=False)
        self.log("train_student_loss_step", student_loss, prog_bar=True, on_epoch=False)
        self.log("clip_train_open_acc_step", clip_train_open_acc, prog_bar=True, on_epoch=False)
        self.log("clip_train_close_acc_step", clip_train_close_acc, prog_bar=True, on_epoch=False)
        self.log("clip_train_total_acc_step", clip_train_total_acc, prog_bar=True, on_epoch=False)
        self.log("train_open_acc_step", train_open_acc, prog_bar=True, on_epoch=False)
        self.log("train_close_acc_step", train_close_acc, prog_bar=True, on_epoch=False)
        self.log("train_total_acc_step", train_total_acc, prog_bar=True, on_epoch=False)

        return {"clip_open_acc": clip_train_open_acc, "clip_close_acc": clip_train_close_acc, "clip_total_acc": clip_train_total_acc,
                "open_acc": train_open_acc, "close_acc": train_close_acc, "total_acc": train_total_acc,
                "teacher_loss": teacher_loss, "student_all_loss": student_all_loss, "student_loss": student_loss,
                "img_loss": img_features_loss, "text_loss": text_features_loss, "fus_loss": fus_features_loss,
                "distillation_loss": distillation_loss}

    def training_epoch_end(self, outputs):
        # outputs 保存了每个step的值
        clip_open_acc = outputs[-1]["clip_open_acc"]  # 最后一个保存的就是平均好的准确率
        clip_close_acc = outputs[-1]["clip_close_acc"]
        clip_total_acc = outputs[-1]["clip_total_acc"]
        open_acc = outputs[-1]["open_acc"]  # 最后一个保存的就是平均好的准确率
        close_acc = outputs[-1]["close_acc"]
        total_acc = outputs[-1]["total_acc"]
        student_all_loss = outputs[-1]["student_all_loss"]  # torch.mean(torch.stack([x["loss"]) for x in output]))
        teacher_loss = outputs[-1]["teacher_loss"]
        student_loss = outputs[-1]["student_loss"]
        img_loss = outputs[-1]["img_loss"]
        text_loss = outputs[-1]["text_loss"]
        fus_loss = outputs[-1]["fus_loss"]
        distillation_loss = outputs[-1]["distillation_loss"]

        self.log("clip_train_open_acc", clip_open_acc, on_step=False, on_epoch=True)
        self.log("clip_train_close_acc", clip_close_acc, on_step=False, on_epoch=True)
        self.log("clip_train_total_acc", clip_total_acc, on_step=False, on_epoch=True)
        self.log("train_open_acc", open_acc, on_step=False, on_epoch=True)
        self.log("train_close_acc", close_acc, on_step=False, on_epoch=True)
        self.log("train_total_acc", total_acc, on_step=False, on_epoch=True)
        self.log("train_student_all_loss", student_all_loss, on_step=False, on_epoch=True)
        self.log("train_teacher_loss", teacher_loss, on_step=False, on_epoch=True)
        self.log("train_student_loss", student_loss, on_step=False, on_epoch=True)
        self.log("train_img_features_loss", img_loss, on_step=False, on_epoch=True)
        self.log("train_text_features_loss", text_loss, on_step=False, on_epoch=True)
        self.log("train_fus_features_loss", fus_loss, on_step=False, on_epoch=True)
        self.log("train_distillation_loss", distillation_loss, on_step=False, on_epoch=True)

        if dist.get_rank() == 0:
            # 将每轮模型的结果保存在json中
            state_dict = {"epoch": self.current_epoch, "train_all_student_loss": float(student_all_loss),
                          "teacher_loss": float(teacher_loss), "student_loss": float(student_loss),
                          "img_loss": float(img_loss), "text_loss": float(text_loss), "fus_loss": float(fus_loss),
                          "distillation_loss": float(distillation_loss),
                          "clip_open_acc": float(clip_open_acc), "clip_close_acc": float(clip_close_acc),
                          "clip_total_acc": float(clip_total_acc), "open_acc": float(open_acc),
                          "close_acc": float(close_acc), "total_acc": float(total_acc)}
            save_epoch_res(self.hparams.args.train_epoch_effect_path, state_dict)

    def shares_validation_code(self, batch, batch_idx):
        image, qus, speech_array, ans, ans_type = batch

        tea_prediction, tea_content = self.teacher_model(image, qus, speech_array)
        stu_prediction, stu_content = self.student_model(image, qus, speech_array)

        tea_img_features, tea_text_features, tea_fus_features = tea_content
        stu_img_features, stu_text_features, stu_fus_features = stu_content

        teacher_loss = self.hard_loss(tea_prediction, ans.view(-1))
        student_loss = self.hard_loss(stu_prediction, ans.view(-1))

        img_features_loss = self.mse_loss(tea_img_features, stu_img_features)
        text_features_loss = self.mse_loss(tea_text_features, stu_text_features)
        fus_features_loss = self.mse_loss(tea_fus_features, stu_fus_features)
        distillation_loss = self.kd_ce_loss(stu_prediction, tea_prediction, self.temp)

        student_all_loss = (1 - self.alpha) * student_loss + self.alpha * (img_features_loss + text_features_loss + fus_features_loss + distillation_loss)

        _, teacher_pred = tea_prediction.max(-1)  # 取出预测值
        _, student_pred = stu_prediction.max(-1)

        # 计算每个batch的准确率
        open_b_acc, close_b_acc, total_b_acc, open_len, close_len, total_b_len = compute_batch_score(teacher_pred, ans.view(-1), ans_type)

        # 计算open、close、total的每个batch后平均精确率
        self.clip_test_e_open_acc += open_b_acc
        self.clip_test_e_open_nums += open_len
        self.clip_test_e_close_acc += close_b_acc
        self.clip_test_e_close_nums += close_len
        self.clip_test_e_total_acc += total_b_acc
        self.clip_test_e_total_nums += total_b_len

        open_b_acc, close_b_acc, total_b_acc, open_len, close_len, total_b_len = compute_batch_score(student_pred, ans.view(-1), ans_type)

        # 计算open、close、total的每个batch后平均精确率
        self.test_e_open_acc += open_b_acc
        self.test_e_open_nums += open_len
        self.test_e_close_acc += close_b_acc
        self.test_e_close_nums += close_len
        self.test_e_total_acc += total_b_acc
        self.test_e_total_nums += total_b_len

        clip_test_open_acc, clip_test_close_acc, clip_test_total_acc = self.shares_step_compute_value(self.clip_test_e_open_acc, self.clip_test_e_close_acc, self.clip_test_e_total_acc, self.clip_test_e_open_nums, self.clip_test_e_close_nums, self.clip_test_e_total_nums)
        test_open_acc, test_close_acc, test_total_acc = self.shares_step_compute_value(self.test_e_open_acc, self.test_e_close_acc, self.test_e_total_acc, self.test_e_open_nums, self.test_e_close_nums, self.test_e_total_nums)

        return clip_test_open_acc, clip_test_close_acc, clip_test_total_acc, test_open_acc, test_close_acc, test_total_acc, teacher_loss, student_loss, student_all_loss, img_features_loss, text_features_loss, fus_features_loss, distillation_loss

    def validation_step(self, batch, batch_idx):
        clip_open_acc, clip_close_acc, clip_total_acc, open_acc, close_acc, total_acc, teacher_loss, student_loss, student_all_loss, img_features_loss, text_features_loss, fus_features_loss, distillation_loss = self.shares_validation_code(batch, batch_idx)

        return {"clip_open_acc": clip_open_acc, "clip_close_acc": clip_close_acc, "clip_total_acc": clip_total_acc,
                "open_acc": open_acc, "close_acc": close_acc, "total_acc": total_acc, "teacher_loss": teacher_loss,
                "student_all_loss": student_all_loss, "student_loss": student_loss,
                "img_loss": img_features_loss, "text_loss": text_features_loss, "fus_loss": fus_features_loss,
                "distillation_loss": distillation_loss}

    def validation_epoch_end(self, outputs):
        # outputs 保存了每个step的值
        clip_open_acc = outputs[-1]["clip_open_acc"]  # 最后一个保存的就是平均好的准确率
        clip_close_acc = outputs[-1]["clip_close_acc"]
        clip_total_acc = outputs[-1]["clip_total_acc"]
        open_acc = outputs[-1]["open_acc"]  # 最后一个保存的就是平均好的准确率
        close_acc = outputs[-1]["close_acc"]
        total_acc = outputs[-1]["total_acc"]
        student_all_loss = outputs[-1]["student_all_loss"]
        teacher_loss = outputs[-1]["teacher_loss"]
        student_loss = outputs[-1]["student_loss"]
        img_loss = outputs[-1]["img_loss"]
        text_loss = outputs[-1]["text_loss"]
        fus_loss = outputs[-1]["fus_loss"]
        distillation_loss = outputs[-1]["distillation_loss"]

        self.log("clip_test_open_acc", clip_open_acc, on_step=False, on_epoch=True)
        self.log("clip_test_close_acc", clip_close_acc, on_step=False, on_epoch=True)
        self.log("clip_test_total_acc", clip_total_acc, on_step=False, on_epoch=True)
        self.log("test_open_acc", open_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_close_acc", close_acc, on_step=False, on_epoch=True)
        self.log("test_total_acc", total_acc, on_step=False, on_epoch=True)
        self.log("test_student_all_loss", student_all_loss, on_step=False, on_epoch=True)
        self.log("test_teacher_loss", teacher_loss, on_step=False, on_epoch=True)
        self.log("test_student_loss", student_loss, on_step=False, on_epoch=True)
        self.log("test_img_features_loss", img_loss, on_step=False, on_epoch=True)
        self.log("test_text_features_loss", text_loss, on_step=False, on_epoch=True)
        self.log("test_fus_features_loss", fus_loss, on_step=False, on_epoch=True)
        self.log("test_distillation_loss", distillation_loss, on_step=False)

        if dist.get_rank() == 0:
            # 将每轮模型的结果保存在json中  float 类型
            state_dict = {"epoch": self.current_epoch, "test_student_all_loss": float(student_all_loss),
                          "teacher_loss": float(teacher_loss), "student_loss": float(student_loss),
                          "img_loss": float(img_loss), "text_loss": float(text_loss), "fus_loss": float(fus_loss),
                          "distillation_loss": float(distillation_loss),
                          "clip_open_acc": float(clip_open_acc), "clip_close_acc": float(clip_close_acc),
                          "clip_total_acc": float(clip_total_acc), "open_acc": float(open_acc),
                          "close_acc": float(close_acc), "total_acc": float(total_acc)}
            save_epoch_res(self.hparams.args.test_epoch_effect_path, state_dict)

    def test_step(self, batch, batch_idx):
        clip_open_acc, clip_close_acc, clip_total_acc, open_acc, close_acc, total_acc, teacher_loss, student_loss, student_all_loss, img_features_loss, text_features_loss, fus_features_loss, distillation_loss = self.shares_validation_code(batch, batch_idx)

        return {"clip_open_acc": clip_open_acc, "clip_close_acc": clip_close_acc, "clip_total_acc": clip_total_acc,
                "open_acc": open_acc, "close_acc": close_acc, "total_acc": total_acc, "teacher_loss": teacher_loss,
                "student_all_loss": student_all_loss, "student_loss": student_loss,
                "img_loss": img_features_loss, "text_loss": text_features_loss, "fus_loss": fus_features_loss,
                "distillation_loss": distillation_loss}

    def test_epoch_end(self, outputs):
        # outputs 保存了每个step的值
        clip_open_acc = outputs[-1]["clip_open_acc"]  # 最后一个保存的就是平均好的准确率
        clip_close_acc = outputs[-1]["clip_close_acc"]
        clip_total_acc = outputs[-1]["clip_total_acc"]
        open_acc = outputs[-1]["open_acc"]  # 最后一个保存的就是平均好的准确率
        close_acc = outputs[-1]["close_acc"]
        total_acc = outputs[-1]["total_acc"]
        student_all_loss = outputs[-1]["student_all_loss"]
        teacher_loss = outputs[-1]["teacher_loss"]
        student_loss = outputs[-1]["student_loss"]
        img_loss = outputs[-1]["img_loss"]
        text_loss = outputs[-1]["text_loss"]
        fus_loss = outputs[-1]["fus_loss"]
        distillation_loss = outputs[-1]["distillation_loss"]

        self.log("clip_test_open_acc", clip_open_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("clip_test_close_acc", clip_close_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("clip_test_total_acc", clip_total_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_open_acc", open_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_close_acc", close_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_total_acc", total_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_student_all_loss", student_all_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_teacher_loss", teacher_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_student_loss", student_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_img_features_loss", img_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_text_features_loss", text_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_fus_features_loss", fus_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_distillation_loss_loss", distillation_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer_teacher = torch.optim.AdamW(self.teacher_model.parameters(), lr=self.hparams.args.tch_learning_rate, weight_decay=self.hparams.args.tch_weights_decay)
        step_lr_teacher = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_teacher, self.hparams.args.epochs)  # 余弦退火

        optimizer_student = torch.optim.AdamW(self.student_model.parameters(), lr=self.hparams.args.std_learning_rate, weight_decay=self.hparams.args.std_weights_decay)
        step_lr_student = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_student, self.hparams.args.epochs)

        optim_dict = [{"optimizer": optimizer_teacher, "lr_scheduler": step_lr_teacher},
                      {"optimizer": optimizer_student, "lr_scheduler": step_lr_student}]

        return optim_dict



