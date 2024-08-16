from model.base import *
from model.eval_model import OneLinearLayer

class UFModel(nn.Module):
    def __init__(self, args, device):
        super(UFModel, self).__init__()
        self.encoder = Encoder_One(args.n_features, args.latent_dim, args, deterministic=True)
        print("encoder = {}".format(self.encoder))
        import sys
        sys.exit()
        self.target_net = Predictor(args.latent_dim, args.y_dim, args.hidden_units, args)
        
        self.args = args
        self.device = device
    
    def forward(self, x): 
        h = self.encoder(x)
        y_pred = self.target_net(h)
        return h, y_pred