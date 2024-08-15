from model.base import *
from model.eval_model import OneLinearLayer

class CFModel(nn.Module):
    def __init__(self, args, device):
        super(CFModel, self).__init__()
        self.encoder = Encoder_One(args.n_features, args.latent_dim, args, deterministic=True)
        self.target_net = Predictor(args.latent_dim, args.y_dim, args.hidden_units, args)
        
        self.args = args
        self.device = device
    
    def forward(self, x): 
        x = x.cpu()
        pattern = [4, 0, 1, 2, 3] 
        indices = []
        for i in range(x.size()[0] // 5):
            indices.extend([indice + i * 5 for indice in pattern])
        
        x_1 = x[indices]
        x_2 = x_1[indices]
        x_3 = x_2[indices]
        x_4 = x_3[indices]
        h_0 = self.encoder(x.cuda())
        h_1 = self.encoder(x_1.cuda())
        h_2 = self.encoder(x_2.cuda())
        h_3 = self.encoder(x_3.cuda())
        h_4 = self.encoder(x_4.cuda())
        h = (h_0 + h_1 + h_2 + h_3 + h_4) / 5
        y_pred = self.target_net(h)
        x = x.cuda()
        #h = self.encoder(x)
        #y_pred = self.target_net(h)
        return h, y_pred