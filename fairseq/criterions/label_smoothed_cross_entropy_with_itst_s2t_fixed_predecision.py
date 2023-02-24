# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import logging
import pdb
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
)

import torch
from fairseq import metrics, utils

logger = logging.getLogger(__name__)


@register_criterion("label_smoothed_cross_entropy_with_itst_s2t_fixed_predecision")
class LabelSmoothedCrossEntropyCriterionWithITSTS2TFixedPredecision(LabelSmoothedCrossEntropyCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size,
        report_accuracy,
        latency_weight_avg,
        latency_weight_avg_type,
        latency_weight_var,
        latency_weight_var_type,
        mass_preservation,
        average_method,
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        from examples.simultaneous_translation.utils.latency import LatencyTraining

        self.eps = label_smoothing
        self.latency_weight_avg = latency_weight_avg
        self.latency_weight_avg_type = latency_weight_avg_type
        self.latency_weight_var = latency_weight_var
        self.latency_weight_var_type = latency_weight_var_type
        self.mass_preservation = mass_preservation
        self.average_method = average_method
        if not (self.latency_weight_avg == float(0.0) and self.latency_weight_var == float(0.0)):
            self.latency_train = LatencyTraining(
                self.latency_weight_avg,
                self.latency_weight_var,
                self.latency_weight_avg_type,
                self.latency_weight_var_type,
                self.mass_preservation,
                self.average_method,
            )
        else:
            self.latency_train = None
            logger.info("LatencyTraining not instantiating; [latency_weight_avg] and [latency_weight_avg] are both 0.")

    @staticmethod
    def add_args(parser):
        super(
            LabelSmoothedCrossEntropyCriterionWithITSTS2TFixedPredecision,
            LabelSmoothedCrossEntropyCriterionWithITSTS2TFixedPredecision,
        ).add_args(parser)
        # fmt: off

        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )
        parser.add_argument(
            "--ignore_prefix_size",
            default=0,
            type=int,
            help="ignore first N tokens",
        )
        parser.add_argument(
            "--report-accuracy",
            default=False,
            type=bool,
            help="report accuracy metric",
        )
        parser.add_argument("--latency-weight-avg", default=0., type=float, metavar='D',
                            help="Average loss weight")
        parser.add_argument("--latency-weight-var", default=0., type=float, metavar='D',
                            help="Variance loss weight")
        parser.add_argument("--latency-weight-avg-type", default="differentiable_average_lagging",
                            help="Statistics for Average loss type")
        parser.add_argument("--latency-weight-var-type", default="variance_delay",
                            help="Statistics for variance loss type")
        parser.add_argument("--average-method", default="weighted_average",
                            help="Average loss type")
        # fmt: on

    def compute_loss(self, model, net_output, sample, reduce=True):
        # Compute cross entropy loss first
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce)
        info_list = [item["p_choose"] for item in net_output[-1]["attn_list"]]
        info = torch.cat(info_list, dim=1)
        bsz, num_heads_x_layers, tgt_len, src_len = info.size()
        target_padding_mask = model.get_targets(sample, net_output).eq(self.padding_idx)
        target_padding_mask = 1 - target_padding_mask.unsqueeze(1).unsqueeze(3).int()

        info_loss = (
            (info / info.sum(dim=-1, keepdim=True))
            * target_padding_mask
            * self.bulid_weight_matrix_s2t(src_len, tgt_len)
        ).sum() / num_heads_x_layers
        norm_loss = torch.dist(
            info * target_padding_mask,
            (info / info.sum(dim=-1, keepdim=True)) * target_padding_mask,
            p=2,
        )

        loss = loss + info_loss + norm_loss

        self.extra_log = {
            "info_loss": info_loss,
            "norm_loss": norm_loss,
        }

        return loss, nll_loss

    def bulid_weight_matrix_s2t(self, src_len, tgt_len):
        r = src_len / tgt_len
        a = torch.arange(1, src_len + 1, device="cuda").unsqueeze(0)
        b = torch.arange(1, tgt_len + 1, device="cuda").unsqueeze(1) * r
        w = torch.abs(a - b) - 1
        w = (w).max(torch.zeros((1, 1), device="cuda"))
        w = w.unsqueeze(0).unsqueeze(1) / (tgt_len * src_len)
        return w

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        info_loss_sum = sum(log.get("info_loss", 0) for log in logging_outputs)
        norm_loss_sum = sum(log.get("norm_loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar("info_loss", info_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar("norm_loss", norm_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg))

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(sum(log.get("n_correct", 0) for log in logging_outputs))
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(meters["n_correct"].sum * 100.0 / meters["total"].sum, 3)
                if meters["total"].sum > 0
                else float("nan"),
            )
