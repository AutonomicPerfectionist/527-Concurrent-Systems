from src import load_cifar10, Net, training_step,evaluate
import torch
from torch.distributed.pipeline.sync.pipe import Pipe
from torch.distributed import rpc
import time
from torchvision.models.vision_transformer import VisionTransformer, Encoder, ConvStemConfig

import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional


class ModelParallelVisionTransformer(VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(torch.nn.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
    ):
        super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout, num_classes, representation_size, norm_layer, conv_stem_configs)
        encoder_layers = list(self.encoder.layers.modules())
        self.encoder_first = torch.nn.Sequential(*(encoder_layers[0:num_layers // 2])).to('cuda:0')
        self.encoder_second = torch.nn.Sequential(*(encoder_layers[num_layers // 2 + 1:])).to('cuda:1')
        self.encoder.ln.to('cuda:1')
        self.encoder.pos_embedding.to('cuda:0')
        self.heads.to('cuda:1')
        self.conv_proj.to('cuda:0')

    def forward(self, x):
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1).to('cuda:0')
        x = torch.cat([batch_class_token, x], dim=1).to('cuda:0')
        x = x + self.encoder.pos_embedding.to('cuda:0')
        x = self.encoder.dropout(x).to('cuda:0')
        x = self.encoder_first(x).to('cuda:1')
        x = self.encoder_second(x).to('cuda:1')
        x = self.encoder.ln(x).to('cuda:1')
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x



if __name__ == '__main__':
    n_epochs = 20
    #model = VisionTransformer(
    #   image_size=32,   # 32x32 images
    #   patch_size=4,    # 4x4 patches
    #   num_classes=10,  # Training on cifar-10
    #   hidden_dim=512,  # Dimensions of the attention heads and therefore the encoder layers
    #   num_layers=6,    # Number of encoder layers
    #   num_heads=8,     # Number of self-attention heads per layer
    #   mlp_dim=512,     # Dimension of the multilayer perceptron
    #   dropout=0.1,
    #   attention_dropout=0.1
    #)

    model = ModelParallelVisionTransformer(
       image_size=32,   # 32x32 images
       patch_size=4,    # 4x4 patches
       num_classes=10,  # Training on cifar-10
       hidden_dim=512,  # Dimensions of the attention heads and therefore the encoder layers
       num_layers=6,    # Number of encoder layers
       num_heads=8,     # Number of self-attention heads per layer
       mlp_dim=512,     # Dimension of the multilayer perceptron
       dropout=0.1,
       attention_dropout=0.1
    )

    print(f"Available devices: {[torch.cuda.device(i) for i in range(torch.cuda.device_count())]}")

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    #model = torch.nn.DataParallel(model)

    #model.to(device)
    trainloader,testloader,classes = load_cifar10(batch_size=1024)
    start = time.time()
    for epoch in range(n_epochs):
        training_step(model, trainloader, epoch, device)
        evaluate(model, testloader)
    print("-"*10,"Training finshed","-"*10)
    print(f"Time: {time.time() - start}")

    start = time.time()
    evaluate(model, testloader)
    print(f"Inference time: {time.time() - start}")
