import torch
from torchvision import transforms

class annet(torch.nn.module):

    def __init__(self, device, ):
        super(annet, self).__init__()

        conv_model = torch.nn.Sequential(
            #64 -> 32 BLOCK
            #CHANNEL SIZE 128
            torch.nn.Conv2d(3,128, 3, padding=1), #Normal convolution layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128,128, 3, padding=1), #Normal convolution layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128,128, 2, stride=2), #Halverende Layer
            torch.nn.ReLU(inplace=True),
            
            #32 -> 16 BLOCK
            #CHANNEL SIZE 256
            torch.nn.Conv2d(128,256, 3, padding=1), #Normal convolution layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256,256, 3, padding=1), #Normal convolution layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256,256, 2, stride=2), #Halverende Layer
            torch.nn.ReLU(inplace=True),
            
            #16 -> 8 BLOCK
            #CHANNEL SIZE 512
            torch.nn.Conv2d(256,512, 3, padding=1), #Normal convolution layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512,512, 3, padding=1), #Normal convolution layer
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(512,512, 2, stride=2), #Halverende Layer
            torch.nn.ReLU(inplace=True),
            
            torch.nn.AdaptiveAvgPool2d((1,1)),
            torch.nn.Flatten(),
        )

        # FCN og X,Y output layer
        self.full_model = torch.nn.Sequential(
            conv_model,
            torch.nn.Linear(512,512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512,512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512,512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512,2), #2 = OUTPUT COORDINATES (x,y screen space)
        )

    def forward(self, input):
        output = self.anNet(input)
        return output

    def preprocess_image(self, input_image):
        preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(torch.tensor(input_image, dtype=torch.float)/255.0)
        return input_tensor.unsqueeze(0)