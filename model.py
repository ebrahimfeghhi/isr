import torch.nn as nn 
import torch

class RNN_feedback(nn.Module):

    """ Vanilla RNN with:
            - Feedback from output
            - Sigmoid nonlinearity over hidden activations 
            - Softmax activation over output 
            - Initialization follows Botvinick and Plaut, 2006 
    """

    def __init__(self, data_size, hidden_size, output_size):

        """ Init model.
        @param data_size (int): Input size
        @param hidden_size (int): the size of hidden states
        @param output_size (int): number of classes
        """
        super(RNN_feedback, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        # recurrent to recurrent connections 
        self.h2h = nn.Linear(hidden_size, hidden_size)
        nn.init.uniform_(self.h2h.weight, -0.5, 0.5)
        nn.init.constant_(self.h2h.bias, -1.0)

        # input to recurrent unit connections 
        self.i2h = nn.Linear(data_size, hidden_size, bias=False)
        nn.init.uniform_(self.i2h.weight, -1.0, 1.0)

        # recurrent unit to output connections 
        self.h2o = nn.Linear(hidden_size, output_size)
        nn.init.uniform_(self.h2o.weight, -1.0, 1.0)
        nn.init.constant_(self.h2o.bias, 0.0)

        # output to recurrent connections 
        self.o2h = nn.Linear(output_size, hidden_size, bias=False)
        nn.init.uniform_(self.o2h.weight, -1.0, 1.0)

        # nonlinearities 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data, last_hidden, last_output):
        """
        @param data: input at time t
        @param last_hidden: hidden state at time t-1
        @param last_output: output at time t-1, y during training and y_hat during testing 
        """
        hidden = self.sigmoid(self.i2h(data) + self.h2h(last_hidden) + self.o2h(last_output))
        output = self.softmax(self.h2o(hidden)) 
        return output, hidden

    def init_hidden_output_state(self, batch_size, device):

        hidden = torch.full((batch_size, self.hidden_size), .5).to(device)
        output = torch.zeros(batch_size, self.output_size).to(device)

        return hidden, output 


class RNN_pytorch(nn.Module):

    def __init__(self, data_size, hidden_size, output_size):

        super(RNN_pytorch, self).__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.RNN = nn.RNN(data_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, data, h0):

        hidden, _ = self.RNN(data, h0)
        output = self.h2o(hidden) 
        return output

class internet_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(internet_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, device):
        # Set initial hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out
