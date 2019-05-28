from pixyz.losses import KullbackLeibler, IterativeLoss, Expectation as E
from pixyz.distributions import Deterministic
from pixyz.models import Model
import torch
from torch import nn

from distribution import Filtering, Transition, Inference, Decoder


class BeliefStateNet(Deterministic):
    def __init__(self, x_size, processed_x_size, b_size):
        super(BeliefStateNet, self).__init__(cond_var=["x"], var=["b"])
        self.fc1 = nn.Linear(x_size, processed_x_size)
        self.fc2 = nn.Linear(processed_x_size, processed_x_size)
        self.lstm = nn.LSTM(input_size=processed_x_size, hidden_size=b_size)
        
    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        b, *_ = self.lstm(h)
        return {"b": b}


class SliceStep(Deterministic):
    def __init__(self):
        super(SliceStep, self).__init__(cond_var=["t", "x", "b"], var=["x_t2", "b_t1", "b_t2"], name="f")

    def forward(self, t, x, b):
        slice_dict = {"x_t2": x[t]}
        slice_dict.update({"b_t1": b[t]})
        slice_dict.update({"b_t2": b[t+1]})
        return slice_dict


class TDVAE(Model):
    def __init__(self, seq_len=16, b_size=50, x_size=1*64*64, processed_x_size=1*64*64, c_size=50, z_size=8,
                 device="cpu", **kwargs):

        # distributions
        self.b_size = b_size
        self.x_size = x_size
        self.processed_x_size = processed_x_size
        self.c_size = c_size
        self.z_size = z_size

        self.p_b1 = Filtering(b_size=self.b_size, z_size=self.z_size).to(device)
        self.p_b2 = self.p_b1.replace_var(b_t1="b_t2", z_t1="z_t2")
        self.p_t = Transition(z_size=self.z_size).to(device)
        self.q = Inference(b_size=self.b_size, z_size=self.z_size).to(device)
        self.p_d = Decoder(x_size=self.x_size, z_size=self.z_size).to(device)
        self.belief_state_net = BeliefStateNet(self.x_size, self.processed_x_size, self.b_size).to(device)
        self.slice_step = SliceStep()

        self.pred_next_step = self.p_t * self.p_b1 * self.slice_step

        # losses
        self.kl = KullbackLeibler(self.q, self.p_b1)
        self.reconst = E(self.q,  -self.p_t.log_prob() - self.p_d.log_prob() + self.p_b2.log_prob())
        self.step_loss = E(self.p_b2, self.reconst + self.kl)

        self._loss = IterativeLoss(self.step_loss, max_iter=seq_len-1,
                                   series_var=["x", "b"], timestep_var=["t"],
                                   slice_step=self.slice_step)
        self.loss = E(self.belief_state_net, self._loss).mean()

        super(TDVAE, self).__init__(loss=self.loss, distributions=[self.p_b1, self.p_b2, self.p_t, self.p_d, self.q,
                                                                   self.belief_state_net],
                                    **kwargs)
    
    def pred(self, batch):
        seq_len, batch_size, C, H, W = batch.size()
        batch = batch.view(seq_len, batch_size, -1)
        samples = self.belief_state_net.sample({"x": batch})

        # prediction
        test_pred = batch.clone()
        for t in range(seq_len-1):
            samples.update({"t": t})
            samples = self.pred_next_step.sample(samples)
            x_t2_hat = self.p_d.sample_mean({"z_t2": samples["z_t2"]})  # (batch_size, C, H, W)
            test_pred[t+1] = x_t2_hat
        test_pred = torch.clamp(test_pred.view(seq_len, batch_size, C, H, W), 0, 1)
        
        return test_pred
