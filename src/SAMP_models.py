import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import numpy as np


def load_model(model_name='MotionNet', **kwargs):
    if model_name == 'MotionNet':
        model = MotionNet
    elif model_name == 'MotionNet_Decoder':
        model = MotionNet_Decoder
    elif model_name == 'GoalNet':
        model = GoalNet
    else:
        err_msg = 'Unknown model name: {}'.format(model_name)
        raise ValueError(err_msg)

    output = model(**kwargs)
    return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class GatingNetwork(torch.nn.Module):
    def __init__(self, rng=None, input_size=None, output_size=None, hidden_size=None, **kwargs):
        super(GatingNetwork, self).__init__()
        self.rng = rng
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.gating_network = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size),
            torch.nn.ELU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
            torch.nn.ELU(),
            torch.nn.Linear(self.hidden_size // 2, self.output_size),
            torch.nn.Softmax(dim=-1))

    def forward(self, inputs):
        return self.gating_network(inputs)


class PredictionNet(torch.nn.Module):
    def __init__(self, rng, num_experts=6, input_size=1664, hidden_size=512, output_size=618, z_dim=32, use_cuda=False,
                 **kwargs):
        super(PredictionNet, self).__init__()
        self.rng = rng
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.z_dim = z_dim
        self.use_cuda = use_cuda

        # Motion network expert parameters
        self.w_l1, self.b_l1 = self.init_params(
            self.num_experts, self.input_size, self.hidden_size)
        self.w_l2, self.b_l2 = self.init_params(
            self.num_experts, self.hidden_size + z_dim, self.hidden_size)
        self.w_l3, self.b_l3 = self.init_params(
            self.num_experts, self.hidden_size + z_dim, self.output_size)

    def init_params(self, num_experts, input_size, output_size):
        w_bound = np.sqrt(6. / np.prod([input_size, output_size]))
        w = np.asarray(
            self.rng.uniform(low=-w_bound, high=w_bound,
                             size=[num_experts, input_size, output_size]),
            dtype=np.float32)
        if self.use_cuda:
            w = torch.nn.Parameter(
                torch.cuda.FloatTensor(w), requires_grad=True)
            b = torch.nn.Parameter(
                torch.cuda.FloatTensor(num_experts, output_size).fill_(0),
                requires_grad=True)
        else:
            w = torch.nn.Parameter(
                torch.FloatTensor(w), requires_grad=True)
            b = torch.nn.Parameter(
                torch.FloatTensor(num_experts, output_size).fill_(0),
                requires_grad=True)
        return w, b

    def dropout_and_linearlayer(self, inputs, weights, bias):
        return torch.sum(inputs[..., None] * weights, dim=1) + bias

    def forward(self, p_prev, blending_coef, z=None):
        # inputs: B*input_dim
        # Blending_coef : B*experts

        w_l1 = torch.sum(
            blending_coef[..., None, None] * self.w_l1[None], dim=1)
        b_l1 = torch.matmul(blending_coef, self.b_l1)

        w_l2 = torch.sum(
            blending_coef[..., None, None] * self.w_l2[None], dim=1)
        b_l2 = torch.matmul(blending_coef, self.b_l2)

        w_l3 = torch.sum(
            blending_coef[..., None, None] * self.w_l3[None], dim=1)
        b_l3 = torch.matmul(blending_coef, self.b_l3)

        h0 = p_prev
        h0 = torch.cat((z, h0), dim=1)

        h1 = F.elu(self.dropout_and_linearlayer(h0, w_l1, b_l1))
        h1 = torch.cat((z, h1), dim=1)

        h2 = F.elu(self.dropout_and_linearlayer(h1, w_l2, b_l2))
        h2 = torch.cat((z, h2), dim=1)

        h3 = self.dropout_and_linearlayer(h2, w_l3, b_l3)
        return h3


class INet(nn.Module):
    def __init__(self, **kwargs):
        super(INet, self).__init__()
        self.I = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU())

    def forward(self, I):
        return self.I(I)


class MotionNet_Encoder(nn.Module):
    def __init__(self, state_dim=5307, z_dim=32, **kwargs):
        super(MotionNet_Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(2 * state_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU())

        self.INet = INet()
        self.fc_mu = nn.Linear(256 * 2, z_dim)
        self.fc_logvar = nn.Linear(256 * 2, z_dim)

    def forward(self, p, p_prev, I=None):
        x = torch.cat((p, p_prev), dim=1)
        x = self.main(x)
        I = self.INet(I)
        x = torch.cat((x, I), dim=1)
        return self.fc_mu(x), self.fc_logvar(x)


class MotionNet_Decoder(nn.Module):
    def __init__(self, state_dim=524, I_enc_dim=256, z_dim=32, h_dim=256, rng=None, num_experts=5, h_dim_gate=256,
                 **kwargs):
        super(MotionNet_Decoder, self).__init__()
        self.INet = INet()
        pred_net_input_dim = state_dim + I_enc_dim + z_dim

        self.gating_network = GatingNetwork(rng=rng, input_size=state_dim + z_dim, output_size=num_experts,
                                            hidden_size=h_dim_gate, **kwargs)
        self.prediction_net = PredictionNet(rng=rng, num_experts=num_experts,
                                            input_size=pred_net_input_dim, hidden_size=h_dim,
                                            output_size=state_dim, z_dim=z_dim, **kwargs)

    def forward(self, z, p_prev, I=None):
        omega = self.gating_network(torch.cat((z, p_prev), dim=1))
        I = self.INet(I)
        x = torch.cat((p_prev, I), dim=1)
        y_hat = self.prediction_net(x, omega, z=z)
        return y_hat


class MotionNet(nn.Module):
    def __init__(self, state_dim=472, **kwargs):
        super(MotionNet, self).__init__()
        self.encoder = MotionNet_Encoder(state_dim=state_dim, **kwargs)
        self.decoder = MotionNet_Decoder(state_dim=state_dim, I_enc_dim=256, **kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, p, p_prev, I=None):
        mu, logvar = self.encoder(p_prev, p, I)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, p_prev, I), mu, logvar


##################################################################
######### GOAL NET ##############################################
##################################################################


class GoalNetEncoder(nn.Module):
    def __init__(self, input_dim, cond_dim, h_dim, z_dim, **kwargs):
        super(GoalNetEncoder, self).__init__()

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 64))

        self.main = nn.Sequential(
            nn.Linear(input_dim + 64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8))

        self.fc1 = nn.Linear(8, z_dim)
        self.fc2 = nn.Linear(8, z_dim)

    def forward(self, x, cond):
        cond = self.cond_encoder(cond)
        x = torch.cat((x, cond), dim=1)
        x = self.main(x)
        return self.fc1(x), self.fc2(x)


class GoalNetDecoder(nn.Module):
    def __init__(self, input_dim, cond_dim, h_dim, z_dim, **kwargs):
        super(GoalNetDecoder, self).__init__()
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 64))

        self.main = nn.Sequential(
            nn.Linear(z_dim + 64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim))

    def forward(self, z, cond):
        cond = self.cond_encoder(cond)
        z = torch.cat((z, cond), dim=1)
        return self.main(z)


class GoalNet(nn.Module):
    def __init__(self, input_dim_goalnet, interaction_dim, h_dim_goalnet, z_dim_goalnet, **kwargs):
        super(GoalNet, self).__init__()
        self.encoder = GoalNetEncoder(input_dim_goalnet, interaction_dim, h_dim_goalnet, z_dim_goalnet)
        self.decoder = GoalNetDecoder(input_dim_goalnet, interaction_dim, h_dim_goalnet, z_dim_goalnet)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, cond), mu, logvar
