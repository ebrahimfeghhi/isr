import torch.nn as nn 
import torch

class RNN_feedback(nn.Module):

    """ Vanilla RNN with:
            - Feedback from output
            - Sigmoid nonlinearity over hidden activations 
            - Softmax activation over output 
            - Initialization follows Botvinick and Plaut, 2006 
    """

    def __init__(self, data_size, hidden_size, output_size,
                feedback_scaling=1, alpha_s=1, alpha_r=1):

        """ Init model.
        @param data_size (int): Input size
        @param hidden_size (int): the size of hidden states
        @param output_size (int): number of classes
        @param init_wrec (string): 'identity' or 'ortho'
        @param alpha_s (float)
        @param alpha_r (float)
        """
        super(RNN_feedback, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fs = feedback_scaling

        self.alpha_s = alpha_s
        self.alpha_r = alpha_r

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

        self.F = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data, x_prev, r_prev, o_prev):
        """
        @param data: input at time t
        @param r_prev: firing rates at time t-1
        @param x_prev: membrane potential values at time t-1
        @param o_prev: output at time t-1
        """
        x = (1-self.alpha_s) * x_prev + self.alpha_s*(self.i2h(data) + self.h2h(r_prev) + self.fs*self.o2h(o_prev))
        r = (1-self.alpha_r)*r_prev + self.alpha_r*self.F(x) 
        output = self.softmax(self.h2o(r)) 
        return output, x, r

    def init_states(self, batch_size, device):

        x0 = torch.full((batch_size, self.hidden_size), 0.5).to(device)
        r0 = torch.full((batch_size, self.hidden_size), 0.5).to(device)
        output = torch.zeros(batch_size, self.output_size).to(device)
        return x0, r0, output


    










    
        




        