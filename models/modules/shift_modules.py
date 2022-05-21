from torch import nn


class Learnable_Shift(nn.Module):
    def __init__(self, input_nc, n_channel, n_convs=2, norm_layer=nn.BatchNorm2d, multiplier=50.0):
        super(Learnable_Shift, self).__init__()
        self.n_channel = n_channel
        self.multiplier = multiplier
        self.model = []
        for i in range(n_convs):
            if i == 0:
                self.model += [nn.Conv2d(input_nc, input_nc, kernel_size=3, padding=1, groups=input_nc), norm_layer(input_nc, eps=1e-04), nn.Conv2d(input_nc, n_channel, kernel_size=1),
                            nn.ReLU(True), norm_layer(n_channel, eps=1e-04),]
            else:
                self.model += [nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1, groups=n_channel), norm_layer(n_channel, eps=1e-04), nn.Conv2d(n_channel, n_channel, kernel_size=1),
                            nn.ReLU(True), norm_layer(n_channel, eps=1e-04)]
        self.model = nn.Sequential(*self.model)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_channel, 2)
        self.tanh = nn.Tanh()
        #self.model += [nn.AvgPool2d(1), nn.Linear(n_channel, 2)]
        

    def forward(self, x):
        x = self.model(x)
        x = self.gap(x)
        x = x.view(-1, self.n_channel)
        output = self.fc(x)
        #print(output)
        output = self.tanh(output) * self.multiplier
        #print(output)
        return output