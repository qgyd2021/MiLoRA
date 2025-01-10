# coding=utf-8
# Copyright 2024 DataSelect AI. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" efficient_mixture of LoRA."""
import random
import time

import numpy as np

import torch
from torch import nn
from torch.distributions.normal import Normal
import torch.nn.functional as F

import sys

from src.llama2.model_utils import weighted_sum

sys.path.append("./")

from src.llama2.rational import Rational


class LoRAModule(nn.Module):
    def __init__(self,
                 input_size=1024,
                 output_size=1024,
                 lora_rank=8,
                 dropout_prob=0.5,
                 lora_alpha=32
                 ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.lora_rank = lora_rank
        self.dropout_prob = dropout_prob
        self.lora_alpha = lora_alpha

        self.module_a = nn.Linear(
            self.input_size,
            self.lora_rank,
            bias=False
        )

        self.module_b = nn.Linear(
            self.lora_rank,
            self.output_size,
            bias=False
        )

    def forward(self, input_tensor=None,):

        a_ = self.module_a(input_tensor) # （bsz, seq_len, lora_rank）
        b_ = self.lora_alpha / self.lora_rank * self.module_b(a_)
        output_tensor = F.dropout(b_, p=self.dropout_prob, training=self.training)
        # c_ = F.dropout(self.module_b(a_), p=self.dropout_prob, training=self.training)
        # output_tensor = lora_gate * c_

        return output_tensor


# class LoRAModuleWithFineGrainedGates(nn.Module):
#     def __init__(self,
#                  input_size=1024,
#                  output_size=1024,
#                  lora_rank=8,
#                  dropout_prob=0.3
#                  ):
#         super().__init__()
#
#         self.input_size = input_size
#         self.output_size = output_size
#         self.lora_rank = lora_rank
#         self.dropout_prob = dropout_prob
#
#         self.module_a = nn.Linear(
#             self.input_size,
#             self.lora_rank,
#             bias=False
#         )
#
#         self.module_b = nn.Linear(
#             self.lora_rank,
#             self.output_size,
#             bias=False
#         )
#
#     def forward(self, input_tensor=None, lora_gate=None):
#
#         a_ = self.module_a(input_tensor) # （bsz, seq_len, lora_rank）
#         # lora_gate: (bsz, lora_rank)
#         bsz = lora_gate.shape[0]
#         lora_rank = lora_gate.shape[-1]
#         lora_gate_new = torch.zeros_like(lora_gate)
#         lora_gate_new = lora_gate_new.unsqueeze(-1).repeat(1, 1, lora_rank)
#         for i in range(bsz):
#             for j in range(lora_rank):
#                 lora_gate_new[i, j, j] = lora_gate[i, j]
#
#         # if self.training:
#         #     print("lora_gate: ", lora_gate.shape)
#         #     print("lora_gate: ", lora_gate)
#         # print("lora_gate_new: ", lora_gate_new)
#         a_ = torch.matmul(a_, lora_gate_new)
#
#         output_tensor = F.dropout(self.module_b(a_), p=self.dropout_prob, training=self.training)
#         # c_ = F.dropout(self.module_b(a_), p=self.dropout_prob, training=self.training)
#         # output_tensor = lora_gate * c_
#
#         return output_tensor


class LoRAGate(nn.Module):
    def __init__(self,
                 input_size=1024,
                 num_experts=7,
                 noisy_gating=True,
                 k=4,
                 dropout_prob=0.3
                 ):
        super().__init__()

        self.input_size = input_size
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.dropout_prob = dropout_prob

        assert (self.k <= self.num_experts)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

        # 参数
        self.w_gate = nn.Parameter(
            torch.zeros(input_size, num_experts), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(input_size, num_experts), requires_grad=True
        )
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))


    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def noisy_top_k_gating(self, x, train, noise_epsilon=2e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ self.w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev)
            # print("noise_stddev: ", noise_stddev)
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            if torch.sum(noisy_logits) == 0.0:
                noisy_logits = torch.randn_like(clean_logits) * 0.1

            logits = noisy_logits

        else:
            # print("eval logits: ", clean_logits)
            # print("eval logits: ", clean_logits)
            logits = clean_logits

        # top_k_pobs, indices = F.softmax(logits, dim=-1).topk(self.k, dim=-1)
        top_k_logits, indices = logits.topk(self.k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        # zeros = torch.full_like(logits, 0.0)
        # router_output = zeros.scatter(-1, indices, top_k_pobs)
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        # print("top_k_logits: ", top_k_logits)
        # print("indices: ", indices)
        # print("sparse_logits: ", sparse_logits)
        if not self.training and random.uniform(0, 1) < 0.002:
            print("router_output: ", router_output)

        top_k_pobs_1, _ = F.softmax(clean_logits, dim=-1).topk(self.k, dim=-1)
        load_loss = torch.sum(top_k_pobs_1)

        # cv_squared = self.cv_squared(F.softmax(clean_logits, dim=-1))
        cv_squared = self.cv_squared(F.softmax(logits, dim=-1))

        return router_output, indices, torch.mean(cv_squared)

    def forward(self, x=None, ):
        """Args:
                x: tensor shape [batch_size, input_size]
                train: a boolean scalar.
                loss_coef: a scalar - multiplier on load-balancing losses

                Returns:
                y: a tensor with shape [batch_size, output_size].
                extra_training_loss: a scalar.  This should be added into the overall
                training loss of the model.  The backpropagation of this loss
                encourages all experts to be approximately equally used across a batch.
        """

        router_output, indices, load_loss = self.noisy_top_k_gating(x, self.training)

        # load balance loss


        return router_output, load_loss


class LastPooler(nn.Module):
    def __init__(self, num_prompt_tokens):
        super().__init__()

        self.num_prompt_tokens = num_prompt_tokens

    def forward(self, input_tensor):
        return input_tensor[:, :, -self.num_prompt_tokens: ]


class SelfAttnPooler(nn.Module):
    """
    A ``SelfAttnAggregator`` is a self_attn layers.  As a
    :class:`Seq2VecEncoder`, the input to this module is of shape ``(batch_size, num_tokens,
    input_dim)``, and the output is of shape ``(batch_size, output_dim)``, where input_dim == output_dim.

    Parameters
    ----------
    """

    def __init__(self, output_dim,) -> None:
        super(SelfAttnPooler, self).__init__()

        self.output_dim = output_dim

        self.attn_vector = nn.Linear(
            self.output_dim,
            1
        )

    def forward(self, input_tensors: torch.Tensor):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        input_tensors : (1, num_tokens, input_dim).
        mask : sentence mask, (1, num_tokens).
        Returns
        -------
        input_self_attn_pooled : torch.FloatTensor
            A tensor of shape ``(batch_size, output_dim)`` .
        """

        input_tensors = input_tensors.transpose(1, 2)
        # print("input_tensors: ", input_tensors.shape)

        # Self-attentive pooling layer
        # Run through linear projection. Shape: (1, sequence length, 1)
        # Then remove the last dimension to get the proper attention shape (1, sequence length).
        self_attentive_logits = self.attn_vector(
            input_tensors
        ).squeeze(2)
        self_weights = torch.softmax(self_attentive_logits, dim=-1)
        # print("self_weights: ", self_weights.shape)
        # print("input_tensors: ", input_tensors.shape)

        input_self_attn_pooled = weighted_sum(input_tensors, self_weights)
        input_self_attn_pooled = input_self_attn_pooled.unsqueeze(1)
        # print("input_self_attn_pooled: ", input_self_attn_pooled.shape)

        input_self_attn_pooled = input_self_attn_pooled.transpose(1, 2)

        return input_self_attn_pooled



class LoRAGateWithPooling(nn.Module):
    def __init__(self,
                 input_size=1024,
                 pooler_type="last",
                 num_experts=7,
                 noisy_gating=True,
                 k=4,
                 dropout_prob=0.3,
                 activation="gelu",
                 ):
        super().__init__()

        self.input_size = input_size
        self.num_experts = num_experts

        self.intermediate_act_fn = None
        print("activation: ", activation)
        if activation == "gelu":
            self.intermediate_act_fn = nn.GELU()
        elif activation == "relu":
            self.intermediate_act_fn = nn.ReLU()
        elif activation == "rational_relu":
            self.intermediate_act_fn = Rational(
                approx_func="relu",
            )
        elif activation == "rational_gelu":
            self.intermediate_act_fn = Rational(
                approx_func="gelu",
            )
        else:
            print("no activation is used ")

        # 池化层
        self.pooler_type = pooler_type
        if self.pooler_type == "avg":
            self.pooler = nn.AdaptiveAvgPool1d(1)
        elif self.pooler_type == "attn":
            self.pooler = SelfAttnPooler(input_size)
        else:
            self.pooler = LastPooler(1)

        # lora gate
        self.lora_gate = LoRAGate(
            input_size=self.input_size,
            num_experts=num_experts,
            noisy_gating=noisy_gating,
            k=k,
            dropout_prob=0.3
        )

        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, hidden_states=None, prompt_se=None, attention_mask=None):
        # prompt_se: 记录soft prompt应该起始的位置
        # print("prompt_se 1: ", prompt_se)

        if self.intermediate_act_fn is not None:
            hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)

        list_lora_gates = []
        # print("hidden_states zxklvkdfmg: ", hidden_states.shape)
        total_gate_loss = 0.0
        for i in range(hidden_states.size(0)):
            hidden_state = hidden_states[i]
            # print("hidden_state 1: ", hidden_state.shape)

            if random.uniform(0, 1) < 0.0001:
                print("hidden_state: ", hidden_state.shape)
                print("prompt_se: ", prompt_se)
                print("i: ", i)

            hidden_state = hidden_state[prompt_se[i][0]: prompt_se[i][1], :].unsqueeze(0)
            hidden_state = hidden_state.transpose(1, 2)  # B x L x D  -->   B x D x L
            hidden_state = (self.pooler(hidden_state)).transpose(1, 2)  # 1 x D
            hidden_state = hidden_state[0, :, :]
            # print("hidden_state 2: ", hidden_state.shape)

            # 计算lora gates
            t0 = time.time()
            gate_, gate_loss = self.lora_gate(hidden_state)
            t1 = time.time()


            total_gate_loss += gate_loss
            # print("gate_: ", gate_)
            # print("gate_ fbvgfdfgg: ", gate_.shape)
            # print("gate_loss: ", gate_loss)
            list_lora_gates.append(gate_)

        lora_gates = torch.concat(list_lora_gates, dim=0)
        total_gate_loss = total_gate_loss / hidden_states.size(0)
        #
        # print("lora_gates  vodvofg: ", lora_gates)
        # print("total_gate_loss  vodvofg: ", total_gate_loss)
        # if not self.training:
        #     print("lora_gates 122: ", lora_gates)

        return lora_gates, total_gate_loss


if __name__ == "__main__":

    batch_size = 1
    sequence_length = 512
    hidden_size = 1024
    x = torch.randn((batch_size, sequence_length, hidden_size))
    x = x.to(torch.device("cuda"))
    prompt_se = torch.tensor(
        np.array([[3, 254]])
    )
    prompt_se = prompt_se.to(torch.device("cuda"))


    lora_gate = LoRAGateWithPooling(
        input_size=hidden_size,
        pooler_type="last",
        num_experts=7,
        noisy_gating=True,
        k=1,
        dropout_prob=0.3,
        activation="gelu"

    )

    lora_gate = lora_gate.to(torch.device("cuda"))

    t0 = time.time()
    for idx in range(1000):
        gates, gate_loss = lora_gate(x, prompt_se)
        # print("gates: ", gates)
        # print("gate_loss: ", gate_loss)

    t1 = time.time()
    print("time cost: ", (t1 - t0) / 1000)





