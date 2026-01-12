import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torchvision.transforms as transforms
import torch
import clip
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

device = "cuda" if torch.cuda.is_available() else "cpu"

# === FORCE float32 ===
model, preprocess = clip.load("ViT-B/32", device=device, download_root="./clip_model/", jit=False)
model = model.float()
model.to(device)
model.eval()  # Set to eval mode
for para in model.parameters():
    para.requires_grad = False

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float32

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class Prompts(nn.Module):
    def __init__(self, initials=None, length_prompt=16):
        super(Prompts, self).__init__()
        print("The initial prompts are:", initials)
        self.text_encoder = TextEncoder(model)
        self.length_prompt = length_prompt
        if isinstance(initials, list):
            text = clip.tokenize(initials).to(device)
            embedding = model.token_embedding(text).float()
            self.embedding_prompt = nn.Parameter(embedding.requires_grad_()).to(device)
        elif isinstance(initials, str):
            prompt_path = initials
            state_dict = torch.load(prompt_path, map_location=device, weights_only=False)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt'].float()).to(device)
            self.embedding_prompt.requires_grad = False  # Don't train prompt during enhancement
        else:
            dummy_text = [" ".join(["X"] * self.length_prompt)] * 2
            text = clip.tokenize(dummy_text).to(device)
            embedding = model.token_embedding(text).float()
            self.embedding_prompt = torch.nn.init.xavier_normal_(nn.Parameter(embedding.requires_grad_())).to(device)

    def forward(self, tensor, flag=1):
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * self.length_prompt)]])
        text_features = self.text_encoder(self.embedding_prompt, tokenized_prompts)
        probs_list = []
        for i in range(tensor.shape[0]):
            image_features = tensor[i].float()
            nor = torch.norm(text_features, dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ (text_features / nor).T)
            if flag == 1:
                probs_list.append(similarity[:, 0].softmax(dim=-1))
            else:
                probs_list.append(similarity)
        return torch.stack(probs_list)

def load_learned_prompt(path, length_prompt=16):
    learn_prompt = Prompts(path).to(device)
    learn_prompt = torch.nn.DataParallel(learn_prompt)
    embedding_prompt = learn_prompt.module.embedding_prompt
    embedding_prompt.requires_grad = False
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * length_prompt)]])
    text_encoder = TextEncoder(model)
    text_features = text_encoder(embedding_prompt, tokenized_prompts)
    return text_features

class L_clip_from_feature(nn.Module):
    def __init__(self):
        super(L_clip_from_feature, self).__init__()
        # Don't register any parameters - we're just using this as a function wrapper
        
    def forward(self, x, text_features):
        """
        Compute CLIP loss between images and learned text features.
        
        FIXED: Ensures gradients flow through to input images
        
        Args:
            x: Images in range [0, 1], shape (B, 3, H, W)
            text_features: Learned text embeddings from prompt (frozen)
            
        Returns:
            Average CLIP similarity score (with gradients!)
        """
        clip_normalizer = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        )
        img_resize = transforms.Resize((224, 224))
        
        # Batch processing for efficiency
        batch_size = x.shape[0]
        
        # Resize and normalize all images at once
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize using CLIP's stats
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
        x_normalized = (x_resized - mean) / std
        
        # Encode images (CLIP model is frozen, but gradients still flow through input)
        # CRITICAL: Don't use torch.no_grad() here!
        image_features = model.encode_image(x_normalized).float()
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_normed = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity (100.0 is CLIP's temperature scaling)
        similarity = (100.0 * image_features @ text_features_normed.T)
        
        # Apply softmax and take mean over batch
        # Use [:, 0] to get first text feature (we only have one learned prompt)
        probs = F.softmax(similarity, dim=-1)[:, 0]
        
        # Return mean score (this will have gradients!)
        return probs.mean()