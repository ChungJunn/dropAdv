import torch
import torch.nn as nn

'''
dim_hidden=8
batch_size=4
drop_p = 0.8

hidden = torch.rand((batch_size, dim_hidden))

p_drop = torch.zeros_like(hidden).type(torch.float32)
p_drop.fill_(drop_p)

m_drop = torch.bernoulli((1.0 - p_drop))
print(m_drop)

# apply mask
out = hidden * m_drop

# adjust scale
out = out * (1.0 / (1.0 - drop_p))
'''

class DropoutNew(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutNew, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, input):
        if not self.training: return input
        else:
            p_drop = torch.zeros_like(input).type(torch.float32)
            p_drop.fill_(self.p)

            m_drop = torch.bernoulli((1.0 - p_drop))

            out = (input * m_drop) * (1.0 / (1.0 - self.p))
            
        return out

    def __repr__(self):
        training_str = ', training:' + str(self.training)
        return self.__class__.__name__ + '(' + 'p=' + str(self.p) + training_str + ')'
