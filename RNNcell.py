import torch.nn as nn 
import torch

class RNNcell(nn.Module):

    """ Vanilla RNN with:
            - Feedback from output
            - Sigmoid nonlinearity over hidden activations 
            - Softmax activation over output 
            - Initialization follows Botvinick and Plaut, 2006 
    """

    def __init__(self, data_size, hidden_size, output_size, feedback_scaling, nonlin='sigmoid'):

        """ Init model.
        @param data_size (int): Input size
        @param hidden_size (int): the size of hidden states
        @param output_size (int): number of classes
        """
        super(RNNcell, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fs = feedback_scaling
        self.nonlin = nonlin

        # recurrent to recurrent connections 
        self.h2h = nn.Linear(hidden_size, hidden_size)
        nn.init.uniform_(self.h2h.weight, -0.5, 0.5)
        nn.init.constant_(self.h2h.bias, -1.0)
        
        # input to recurrent unit connections 
        self.i2h = nn.Linear(data_size, hidden_size, bias=False)
        nn.init.uniform_(self.i2h.weight, -1.0, 1.0)

        # output to recurrent connections 
        self.o2h = nn.Linear(output_size, hidden_size, bias=False)
        nn.init.uniform_(self.o2h.weight, self.fs*-1.0, self.fs*1.0)

        if nonlin == 'sigmoid':
            self.F = nn.Sigmoid()
        if nonlin == 'relu':
            self.F = nn.ReLU()
        if nonlin == 'tanh':
            self.F = nn.Tanh()
        if nonlin == 'linear':
            self.F = nn.Identity()

    def forward(self, data, h_prev, o_prev):
        """
        @param data: input at time t
        @param r_prev: firing rates at time t-1
        @param x_prev: membrane potential values at time t-1
        @param o_prev: output at time t-1
        """
        h = self.F((self.i2h(data) + self.h2h(h_prev) + self.o2h(o_prev)))
        return h

class RNN_one_layer(nn.Module):

    """ Multilayer RNN """

    def __init__(self, input_size, hidden_size, output_size, feedback_scaling, nonlin='sigmoid'):

        """ Init model.
        @param data_size: Input size
        @param hidden_size: the size of hidden states
        @param output_size: number of classes
        """
        super(RNN_one_layer, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.RNN = RNNcell(input_size, hidden_size, output_size, feedback_scaling, nonlin)

        self.h2o = nn.Linear(hidden_size, output_size)
        nn.init.uniform_(self.h2o.weight, -1.0, 1.0)
        nn.init.uniform_(self.h2o.bias, -1.0, 1.0)

    def forward(self, data, h_prev, o_prev):
        """
        @param data: input at time t
        @param h_prev : firing rates at time t-1 
        @param o_prev: output at time t-1
        """

        h = self.RNN(data, h_prev, o_prev)

        output = self.h2o(h)

        return output, h

    def init_states(self, batch_size, device):

        h0 = torch.full((batch_size, self.hidden_size), .5).to(device)
        output = torch.zeros(batch_size, self.output_size).to(device)

        return output, h0

class RNN_two_layers(nn.Module):

    """ Multilayer RNN """

    def __init__(self, input_size, hidden_size, output_size, feedback_scaling, nonlin, fb_type):

        """ Init model.
        @param data_size: Input size
        @param hidden_size (list): the size of hidden states
        @param output_size: number of classes
        @param feedback_scaling: scaling factor applied to o2h weights
        @param loss_fn: kl or ce
        @param nonlin: sigmoid, relu, tanh, or linear
        @param fb_type: feedback_type, 0 for feedback from h2 and output to h1, and 1 for feedback only from h2 to h1.
        h2 receives feedback from output in both cases. 
        """
        super(RNN_two_layers, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fb_type = fb_type
        
        self.num_layers = len(hidden_size)

        if fb_type == 0:
            self.RNN1 = RNNcell(input_size, hidden_size[0], hidden_size[1]+output_size, feedback_scaling[0], nonlin)
            self.RNN2 = RNNcell(hidden_size[0], hidden_size[1], output_size, feedback_scaling[1], nonlin)
        if fb_type == 1:
            self.RNN1 = RNNcell(input_size, hidden_size[0], hidden_size[1], feedback_scaling[0], nonlin)
            self.RNN2 = RNNcell(hidden_size[0], hidden_size[1], output_size, feedback_scaling[1], nonlin)

        self.h2o = nn.Linear(hidden_size[1], output_size)
        nn.init.uniform_(self.h2o.weight, -1.0, 1.0)
        nn.init.uniform_(self.h2o.bias, -1.0, 1.0)

    def forward(self, data, h_prev, o_prev):
        """
        @param data: input at time t for first layer
        @param h_prev (list): firing rates at time t-1 for both layers 
        @param o_prev tensor: output at time t-1 for last layer
        """
        if self.fb_type == 0:
            h1 = self.RNN1(data, h_prev[0], torch.cat((h_prev[1], o_prev),dim=1))
        elif self.fb_type == 1:
            h1 = self.RNN1(data, h_prev[0], h_prev[1])

        h2 = self.RNN2(h1, h_prev[1], o_prev)
        output = self.h2o(h2)

        return output, [h1, h2]

    def init_states(self, batch_size, device):

        h0_1 = torch.full((batch_size, self.hidden_size[0]), .5).to(device)
        h0_2 = torch.full((batch_size, self.hidden_size[1]), .5).to(device)
        output = torch.zeros(batch_size, self.output_size).to(device)

        return output, [h0_1, h0_2]
