import torch
from torch import nn

class Conv2dLSTMCell(nn.Module):
    """
    2d convolutional long short-term memory (LSTM) cell.
    Functionally equivalent to nn.LSTMCell with the
    difference being that nn.Kinear layers are replaced
    by nn.Conv2D layers.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of image kernel
    :param stride: length of kernel stride
    :param padding: number of pixels to pad with
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv2dLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kwargs = dict(kernel_size=kernel_size, stride=stride, padding=padding)

        self.forget = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.input  = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.output = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.state  = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, input, states):
        """
        Send input through the cell.

        :param input: input to send through
        :param states: (hidden, cell) pair of internal state
        :return new (hidden, cell) pair
        """
        (hidden, cell) = states

        forget_gate = torch.sigmoid(self.forget(input))
        input_gate  = torch.sigmoid(self.input(input))
        output_gate = torch.sigmoid(self.output(input))
        state_gate  = torch.tanh(self.state(input))

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell

