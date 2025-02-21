import torch
import torchvision
from torchvision import transforms as transforms
import torch.nn.functional as F
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)
    elif (weights == "tiny_vit"):
        return get_tiny_vit(name=name, **kwargs)
    elif (weights == "deit"):
        return get_deit(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def get_tiny_vit(name, **kwargs):
    model = TinyViTFeatureExtractor()

    model = model.eval()
    return model


def get_deit(model_name='facebook/deit-tiny-patch16-224', **kwargs):
    model = DeiTFeatureExtractorModel(model_name=model_name)
    model = model.eval()
    return model

class TinyViTFeatureExtractor(torch.nn.Module):
    def __init__(self, model_name='tiny_vit_5m_224.dist_in22k_ft_in1k', return_attentions=False):
        super(TinyViTFeatureExtractor, self).__init__()
        import timm

        # Create the model with timm
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0  # remove classifier nn.Linear
        )
        # Resolve the data configuration for the model
        self.data_config = timm.data.resolve_model_data_config(self.model)
        # Create the transformation function for the model
        # self.transforms = timm.data.create_transform(**self.data_config, is_training=False)
        self.transforms_tensor = create_transform_tensor(self.data_config)

        self.model.eval()  # Set the model to evaluation mode
        
        self.return_attentions = return_attentions

        # # Define the output shape based on the model's feature dimension
        # self.output_shape = self.model.num_features

    def forward(self, image):
        # Apply transforms
        input_tensor = self.transforms_tensor(image)  # Add batch dimension
        # Extract features
        with torch.no_grad():
            features = self.model.forward_features(input_tensor)
            features = self.model.forward_head(features)
            if not self.return_attentions:
                return features
            else:
                attentions = None
                return features, attentions


class DeiTFeatureExtractorModel(torch.nn.Module):
    def __init__(self, model_name='facebook/deit-tiny-patch16-224', return_attentions=False, do_rescale=True):
        super(DeiTFeatureExtractorModel, self).__init__()
        from transformers import DeiTFeatureExtractor, DeiTModel, DeiTConfig

        # Load the DeiT feature extractor and model
        self.feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
        
        config = DeiTConfig.from_pretrained(model_name)
        config.output_attentions = return_attentions  # Ensure attentions are outputted
       
        self.model = DeiTModel.from_pretrained(model_name, config=config)
        self.model.eval()  # Set the model to evaluation mode
        
        self.return_attentions = return_attentions
        self.do_rescale = do_rescale

    def forward(self, image):
        # If the input images have pixel values between 0 and 1, set do_rescale=False to avoid rescaling them again
        if self.do_rescale:
            inputs = self.feature_extractor(images=image, return_tensors="pt")
        else:
            inputs = self.feature_extractor(images=image, return_tensors="pt", do_rescale=False)
        
        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Extract feature embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        if not self.return_attentions:
            return embeddings
        else:
            attentions = outputs.attentions  # Get attention weights
            return embeddings, attentions



def create_transform_tensor(data_config):
    # Map interpolation string to torchvision interpolation method
    interpolation_methods = {
        'bicubic': 'bicubic',
        'bilinear': 'bilinear',
        'nearest': 'nearest'
    }

    # Extract parameters from data_config
    input_size = data_config['input_size']
    resize_size = int(input_size[1] / data_config['crop_pct'])  # Assuming input_size is (C, H, W)
    interpolation = interpolation_methods.get(data_config['interpolation'], 'bicubic')
    mean = data_config['mean']
    std = data_config['std']

    # Define the transformations for tensor images
    def transform_tensor(img_tensor):
        # Check if the input is a batch of images or a single image
        if img_tensor.ndim == 4:
            batch_mode = True
            batch_size, channels, height, width = img_tensor.size()
        else:
            batch_mode = False
            channels, height, width = img_tensor.size()

        # Resize the tensor image
        img_tensor = F.interpolate(img_tensor.unsqueeze(0) if not batch_mode else img_tensor, size=resize_size, mode=interpolation, align_corners=False)
        if not batch_mode:
            img_tensor = img_tensor.squeeze(0)

        # Center crop the tensor image
        if batch_mode:
            _, h, w = img_tensor.size(1), img_tensor.size(2), img_tensor.size(3)
        else:
            _, h, w = img_tensor.size(0), img_tensor.size(1), img_tensor.size(2)
        th, tw = input_size[1], input_size[2]
        i = (h - th) // 2
        j = (w - tw) // 2
        img_tensor = img_tensor[:, :, i:i+th, j:j+tw] if batch_mode else img_tensor[:, i:i+th, j:j+tw]

        # Normalize the tensor image
        normalize = transforms.Normalize(mean=mean, std=std)
        img_tensor = normalize(img_tensor) if not batch_mode else torch.stack([normalize(img_tensor[b]) for b in range(batch_size)])
        
        return img_tensor
    
    return transform_tensor


def plot_attention_map(attention, input_image):
    # Normalize the attention map
    attention = (attention - attention.min()) / (attention.max() - attention.min())
    
    # Ensure attention map has a batch and channel dimension
    attention = torch.tensor(attention).unsqueeze(0).unsqueeze(0)
    
    # Resize the attention map to the input image size
    attention = F.interpolate(attention, size=(input_image.shape[1], input_image.shape[2]), mode='bilinear').squeeze().cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    if isinstance(input_image, torch.Tensor):
        input_image = transforms.ToPILImage()(input_image)
    ax[0].imshow(input_image)
    ax[0].axis('off')
    ax[0].set_title('Original Image')
    ax[1].imshow(input_image)
    ax[1].imshow(attention, cmap='viridis', alpha=0.6)
    ax[1].axis('off')
    ax[1].set_title('Attention Map Overlay')
    plt.show()

def test_example_img(model_type='deit'):
    img = Image.open(urlopen(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    )).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'deit':
        model = DeiTFeatureExtractorModel(model_name='facebook/deit-tiny-patch16-224', return_attentions=True, do_rescale=False)
    elif model_type == 'tiny_vit':
        model = TinyViTFeatureExtractor(model_name='tiny_vit_5m_224.dist_in22k_ft_in1k', return_attentions=True)
    else:
        raise ValueError("Unsupported model type. Choose 'deit' or 'tiny_vit'.")

    model.to(device)
    img_tensor = img_tensor.to(device)

    if model_type == 'deit':
        embeddings, attentions = model(img_tensor)
    elif model_type == 'tiny_vit':
        embeddings, attentions = model(img_tensor)

    # Print diagnostics
    print(f'Image tensor shape: {img_tensor.shape}')
    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Attention shape: {[att.shape for att in attentions]}')

    # Aggregate attention maps across layers and heads
    aggregated_attention = torch.mean(torch.stack(attentions), dim=(0, 1, 2)).squeeze().cpu().numpy()
    print(f'Aggregated attention shape: {aggregated_attention.shape}')
    
    # Plot the aggregated attention map
    plot_attention_map(aggregated_attention, img_tensor.squeeze())

if __name__ == '__main__':
    # Test with DeiT model
    # test_example_img(model_type='deit')

    # Test with TinyViT model
    test_example_img(model_type='tiny_vit')
