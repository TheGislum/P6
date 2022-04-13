from torch import nn

class annetV2(nn.Module):

    def __init__(self, device, in_channels=3):
        super(annetV2, self).__init__()

        conv_model = nn.Sequential(
            #64 -> 32 BLOCK
            #CHANNEL SIZE 128
            nn.Conv2d(in_channels,128, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, stride=2, padding=1), #Halverende Layer
            nn.ReLU(inplace=True),
            
            #32 -> 16 BLOCK
            #CHANNEL SIZE 256
            nn.Conv2d(128,256, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, stride=2, padding=1), #Halverende Layer
            nn.ReLU(inplace=True),
            
            #16 -> 8 BLOCK
            #CHANNEL SIZE 512
            nn.Conv2d(256,512, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, stride=2, padding=1), #Halverende Layer
            nn.ReLU(inplace=True),
            
            #8 -> 4 BLOCK
            #CHANNEL SIZE 512
            nn.Conv2d(512,512, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, stride=2, padding=1), #Halverende Layer
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

        # FCN og X,Y output layer
        self.fcn_model = nn.Sequential(
            conv_model,
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,2), #2 = OUTPUT COORDINATES (x,y screen space)
            nn.Tanh(),
        ).to(device)

    def forward(self, input):
        output = self.full_model(input)
        return output


class annetV1(nn.Module):

    def __init__(self, device, in_channels=3):
        super(annetV1, self).__init__()

        conv_model = nn.Sequential(
            #64 -> 32 BLOCK
            #CHANNEL SIZE 128
            nn.Conv2d(in_channels,128, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, 2, stride=2), #Halverende Layer
            nn.ReLU(inplace=True),
            
            #32 -> 16 BLOCK
            #CHANNEL SIZE 256
            nn.Conv2d(128,256, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, 2, stride=2), #Halverende Layer
            nn.ReLU(inplace=True),
            
            #16 -> 8 BLOCK
            #CHANNEL SIZE 512
            nn.Conv2d(256,512, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 3, padding=1), #Normal convolution layer
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, 2, stride=2), #Halverende Layer
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
        )

        # FCN og X,Y output layer
        self.full_model = nn.Sequential(
            conv_model,
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,2), #2 = OUTPUT COORDINATES (x,y screen space)
        ).to(device)

    def forward(self, input):
        output = self.full_model(input)
        return output