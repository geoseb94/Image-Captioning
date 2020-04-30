import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        # save params
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        # define layers
        self.w_embeddings = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 1024)    # 1st fully connected layer
        self.linear2 = nn.Linear(1024, vocab_size)     # 2nd fully connected layer  - 2 layer upsampling approach
        self.dropout = nn.Dropout(p=0.3)               # to avoid overfitting
    
    def forward(self, features, captions):
        captions = self.w_embeddings(captions[:,:-1])
        embedded = torch.cat((features.unsqueeze(1), captions), dim=1)

        lstm_out, _ = self.lstm(embedded)
        outputs = self.linear1(lstm_out)
        outputs = self.dropout(outputs)
        outputs = self.linear2(outputs)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Accepts pre-processed image tensor (inputs) and returns predicted sentence 
        (list of tensor ids of length max_len) 
        """
        res = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)         
            outputs = self.linear1(lstm_out.squeeze(1))       
            outputs = self.linear2(outputs) 
            _, predicted = outputs.max(dim=1)                    
            res.append(predicted.item())
            
            inputs = self.w_embeddings(predicted)             
            inputs = inputs.unsqueeze(1)                       
        return res