import torch.nn as nn 
import torch

class LSTMCELL(nn.Module):

    def __init__(self, data_size, hidden_size, output_size, feedback):
        super(LSTMCELL, self).__init__()
        self.data_size = data_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.feedback = feedback

        self.i2h = nn.Linear(data_size, hidden_size)
        #nn.init.uniform_(self.i2h.weight, -1.0, 1.0) 

        self.h2i = nn.Linear(hidden_size, hidden_size, bias=False)
        #nn.init.uniform_(self.h2i.weight, -1.0, 1.0) 

        self.i2f = nn.Linear(data_size, hidden_size)
        #nn.init.uniform_(self.i2f.weight, -1.0, 1.0) 

        self.h2f = nn.Linear(hidden_size, hidden_size, bias=False)
        #nn.init.uniform_(self.h2f.weight, -1.0, 1.0) 

        self.i2g = nn.Linear(data_size, hidden_size)
        #nn.init.uniform_(self.i2g.weight, -1.0, 1.0) 

        self.h2g = nn.Linear(hidden_size, hidden_size, bias=False)
        #nn.init.uniform_(self.h2g.weight, -1.0, 1.0) 

        self.i2o = nn.Linear(data_size, hidden_size)
        #nn.init.uniform_(self.i2o.weight, -1.0, 1.0) 

        self.h2o = nn.Linear(hidden_size, hidden_size, bias=False)
        #nn.init.uniform_(self.h2o.weight, -1.0, 1.0) 

        self.h2output = nn.Linear(hidden_size, output_size)
        #nn.init.uniform_(self.h2output.weight, -1.0, 1.0) 

        if self.feedback:
            self.output2c = nn.Linear(hidden_size, output_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, data, h_prev, c_prev, o_prev):

        i_t = self.sigmoid(self.i2h(data) + self.h2i(h_prev)) # input gate
        f_t = self.sigmoid(self.i2f(data) + self.h2f(h_prev)) # forget gate
        g_t = self.tanh(self.i2g(data) + self.h2g(h_prev)) # input gate
        o_t = self.sigmoid(self.i2o(data) + self.h2o(h_prev)) # output gate 

        # cell state
        if self.feedback:
            c_t = f_t * c_prev + i_t * g_t + self.output2c(o_prev) 
        else:
            c_t = f_t * c_prev + i_t * g_t 
            
        h_t = o_t * self.tanh(c_t) # hidden state

        output = self.softmax(self.h2output(h_t))

        return output, h_t, c_t 

    def init_states(self, batch_size, device):

        y0 = torch.full((batch_size, self.hidden_size), 0.0).to(device)
        c0 = torch.full((batch_size, self.hidden_size), 0.0).to(device)
        h0 = torch.full((batch_size, self.hidden_size), 0.0).to(device)
        return h0, c0, y0
