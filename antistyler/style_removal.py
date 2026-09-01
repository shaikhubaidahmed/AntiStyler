import torch
import torch.nn as nn
import torchvision.models as models

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.MSELoss()(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.gram_matrix(input)
        self.loss = nn.MSELoss()(G, self.target)
        return input

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class AntiStylerStyleRemoval:
    """
    Implements the AntiStyler style-removal procedure according to the experimental contract.
    """
    def __init__(self, config: dict, device: torch.device):
        self.device = device
        self.content_layers = config.get("content_layer", ["conv_4"])
        self.style_layers = config.get("style_layers", ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"])
        self.alpha = config.get("alpha", 1.0)
        self.beta = config.get("beta", 1000.0)
        self.optimization_steps = config.get("optimization_steps", 1)
        self.learning_rate = config.get("learning_rate", 0.05)
        self.padding_size = config.get("padding_size", 10)
        self.style_distribution = config.get("style_distribution", "uniform")
        self.vgg19_weights_path = config.get("vgg19_weights", None)
        
        # NOTEBOOK fallback: no explicit normalization is used.
        try:
            if self.vgg19_weights_path:
                import os
                if not os.path.exists(self.vgg19_weights_path):
                    raise FileNotFoundError(f"Provided VGG19 checkpoint {self.vgg19_weights_path} does not exist.")
                self.cnn = models.vgg19(weights=None)
                state_dict = torch.load(self.vgg19_weights_path, map_location=self.device)
                self.cnn.load_state_dict(state_dict)
                self.cnn = self.cnn.features.eval().to(self.device)
            else:
                self.cnn = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features.eval().to(self.device)
        except Exception as e:
            raise RuntimeError("VGG19 PRETRAINED WEIGHTS NOT AVAILABLE LOCALLY") from e
            
    def _get_style_model_and_losses(self, cnn, style_img, content_img):
        cnn = cnn.eval()
        content_losses = []
        style_losses = []

        model = nn.Sequential()
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = f'conv_{i}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{i}'
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{i}'
            elif isinstance(layer, nn.BatchNorm2d):
                name = f'bn_{i}'
            else:
                raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module(f"content_loss_{i}", content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module(f"style_loss_{i}", style_loss)
                style_losses.append(style_loss)

        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break
        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def remove_style(self, padded_input: torch.Tensor, seed: int = None) -> tuple:
        """
        Removes style from padded input.
        Returns:
            anti_styled_image: (B, C, padded_H, padded_W) float tensor [0,1] RGB
            padded_input: (B, C, padded_H, padded_W) original padded input passed back for convenience
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        anti_styled_image = padded_input.clone().requires_grad_(True)
        
        # Create random style image matching PAPER
        if self.style_distribution == "uniform":
            style_image = torch.rand_like(padded_input)
        else:
            style_image = torch.randn_like(padded_input)
            
        model, style_losses, content_losses = self._get_style_model_and_losses(self.cnn, style_image, padded_input)
        
        optimizer = torch.optim.Adam([anti_styled_image], lr=self.learning_rate)
        
        for step in range(self.optimization_steps):
            def closure():
                with torch.no_grad():
                    anti_styled_image.clamp_(0, 1)

                optimizer.zero_grad()
                model(anti_styled_image)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # SR minimizes content loss and maximizes style loss
                # L_SR = alpha * L_C - beta * L_S
                loss = self.alpha * content_score - self.beta * style_score
                loss.backward()

                return loss

            optimizer.step(closure)

        with torch.no_grad():
            anti_styled_image.clamp_(0, 1)

        return padded_input, anti_styled_image
