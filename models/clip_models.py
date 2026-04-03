from .clip import clip
import torch.nn as nn
import torch
import torch.nn.functional as F
import models.networks.customnet as customnetworks
from models.clip.model import ResidualAttentionBlock
from models.NoiseView.NoiseView import DCTHighPass, DWTHighPass, SpatialFrequencyFusion, CBAMFusion, GroupCBAMEnhancer, GroupSpatialFrequencyFusion
from models.decoder.ASPP import CLIP_ASPP_Adapter
from models.decoder.ConPRN import ContrastiveRPN
from models.decoder.detector import ForgeryDetector
import re

# Model for localisation
class CLIPModelLocalisation(nn.Module):
    def __init__(self, name, intermidiate_layer_output = None, decoder_type = "conv-4", use_noise_view = False, noise_extractor = "dct", use_noise_guided_amplification = False, use_aspp = False, use_conprn = False, use_simdet = False):
        super(CLIPModelLocalisation, self).__init__()
        
        self.intermidiate_layer_output = intermidiate_layer_output
        self.decoder_type = decoder_type
        self.name = name # architecure
        
        if self.intermidiate_layer_output:
            assert "layer" in self.intermidiate_layer_output or "all" in self.intermidiate_layer_output or "xceptionnet" in self.intermidiate_layer_output

        self._set_backbone()
        self._set_decoder()
        # ADD: Noise View
        self.use_noise_view = use_noise_view
        self.noise_extractor_name = noise_extractor
        self.use_noise_guided_amplification = use_noise_guided_amplification
        self._last_noise_map = None
        self._set_noise_view(self.use_noise_view)
        # ADD: ASPP Decoder
        self.use_aspp = use_aspp
        if self.use_aspp:
            self.aspp_adapter = CLIP_ASPP_Adapter()
        # ADD: Contrastive RPN
        self.use_conprn = use_conprn
        if self.use_conprn:
            self.label = None
            self.con_loss = 0.0
            self.conprn = ContrastiveRPN()
        # ADD: SimDet
        self.use_simdet = use_simdet

        if self.use_simdet:
            self.detector = ForgeryDetector(feature_dim=1024, hidden_dim=512, dropout=0.5)
            self.det_pred_probs = None
            # self.det_loss = 0.0
            # self.det_metrics = {
            #     'f1': None,
            #     'auc': None,
            #     'mcc': None
            # }


    def _set_backbone(self):    
        # Set up the backbone model architecture and parameters
        if self.name in ["RN50", "ViT-L/14"]:
            self.model, self.preprocess = clip.load(self.name, device="cpu", intermidiate_layers = (self.intermidiate_layer_output != None))            
        elif self.name == "xceptionnet":
            # XceptionNet
            layername = 'block%d' % 2
            extra_output = 'block%d' % 1
            self.model = customnetworks.make_patch_xceptionnet(
                layername=layername, extra_output=extra_output, num_classes=2)
        # ViT+RN fusion
        elif "RN50" in self.name and "ViT-L/14" in self.name:
            name = self.name.split(",")
            model1, self.preprocess = clip.load(name[0], device="cpu", intermidiate_layers = (self.intermidiate_layer_output != None)) 
            model2, self.preprocess = clip.load(name[1], device="cpu", intermidiate_layers = (self.intermidiate_layer_output != None))            
            self.model = [model1.to("cuda"), model2.to("cuda")]
        elif 'GeoRSCLIP' in self.name:
            model_path = '/irip/fanziyu_bishe/data/Basemodel/GeoRSCLIP/ckpt/RS5M_ViT-L-14.pt'
            self.model, self.preprocess = clip.load(model_path, device="cpu", intermidiate_layers = (self.intermidiate_layer_output != None))
        
    def _set_decoder(self):
        # Set up decoder architecture
        upscaling_layers = []
        if "conv" in self.decoder_type:
            filter_sizes = self._get_conv_filter_sizes(self.name, self.intermidiate_layer_output, self.decoder_type)
            num_convs = int(re.search(r'\d{0,3}$', self.decoder_type).group())
            
            for i in range(1, len(filter_sizes)):
                upscaling_layers.append(nn.Conv2d(filter_sizes[i-1], filter_sizes[i], kernel_size=5, padding=2))
                upscaling_layers.append(nn.BatchNorm2d(filter_sizes[i]))
                upscaling_layers.append(nn.ReLU())
                for _ in range(num_convs//4 - 1):
                    upscaling_layers.append(nn.Conv2d(filter_sizes[i], filter_sizes[i], kernel_size=5, padding=2))
                    upscaling_layers.append(nn.BatchNorm2d(filter_sizes[i]))
                    upscaling_layers.append(nn.ReLU())

                # skip some upscaling layers if the input is too large (case for CNNs)
                skip_upscaling = (
                    self.intermidiate_layer_output == "layer2" and i == 1
                    or self.intermidiate_layer_output == "layer1" and i <= 2
                    ) and ("RN50" in self.name or "xceptionnet" in self.name)
                if skip_upscaling:
                    continue

                upscaling_layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))

            # CNNs output may not be in (256, 256) - usually a (224, 224) size
            if "RN50" in self.name or "xceptionnet" in self.name:
                upscaling_layers.append(nn.Upsample(size=(256, 256), mode='bilinear'))

            upscaling_layers.append(nn.Conv2d(64, 1, kernel_size=5, padding=2))

        elif self.decoder_type == "linear":
            # Xceptionnet
            if self.name == "xceptionnet":
                upscaling_layers.append(nn.Linear(784, 1))
            # CLIP
            else:
                upscaling_layers.append(nn.Linear(1024, 1))

        elif self.decoder_type == "attention":
            transformer_width = 1024
            transformer_heads = transformer_width // 64
            attn_mask = self._build_attention_mask()
            self.att1 = ResidualAttentionBlock(transformer_width, transformer_heads, attn_mask)
            self.att2 = ResidualAttentionBlock(transformer_width, transformer_heads, attn_mask)
            upscaling_layers.append(nn.Linear(1024, 1))

        self.fc = nn.Sequential(*upscaling_layers)

    #ADD
    def _set_noise_view(self, use_noise_view):
        print(f"Using {use_noise_view} noise view")
        if self.noise_extractor_name == "dwt":
            self.noise_extractor = DWTHighPass(levels=2)
        else:
            self.noise_extractor = DCTHighPass(kernel_size=8)
        if use_noise_view == "light":
            self.noise_fusion = SpatialFrequencyFusion()
        elif use_noise_view == "cbam":
            self.noise_fusion = CBAMFusion()
        elif use_noise_view == "group":
            # self.noise_fusion = GroupCBAMEnhancer()
            self.noise_fusion = GroupSpatialFrequencyFusion(num_groups=8)
            

    def _get_conv_filter_sizes(self, name, intermidiate_layer_output, decoder_type):
        assert "conv" in decoder_type

        if "RN50" in name and "ViT-L/14" in name:
            num_layers = len(name.split(","))
            return [1024*num_layers, 512, 256, 128, 64]
        elif "RN50" in name:
            if intermidiate_layer_output == "layer1":
                return [256, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer2":
                return [512, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer3":
                return [1024, 512, 256, 128, 64]
            elif intermidiate_layer_output == "layer4":
                return [2048, 512, 256, 128, 64]
        elif "xceptionnet" in name:
            return [256, 512, 256, 128, 64]
        else:
            return [1024, 512, 256, 128, 64]
    
    def _unify_linear_layer_outputs(self, linear_outputs):
        output = torch.cat(linear_outputs, dim=1)
        output = output.view(output.size()[0],  int(output.size()[1]**0.5), int(output.size()[1]**0.5))
        output = torch.nn.functional.interpolate(output.unsqueeze(1), size = (256, 256), mode = 'bicubic')
        return output

    # standard CLIPs method
    def _build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        context_length = 257
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def _feature_map_transform(self, input):
        output = input.permute(1, 2, 0)
        output = output.view(output.size()[0], output.size()[1], int(output.size()[2]**0.5), int(output.size()[2]**0.5))
        return output

    #new
    def _noise_guided_topk_attention(self, tokens, noise_features):
        bsz, num_tokens, _ = tokens.shape
        if num_tokens <= 1:
            return tokens

        k = min(max(4, num_tokens // 8), 32, num_tokens - 1)
        tokens_fp32 = tokens.float()
        noise_features = noise_features.float()
        noise_features = noise_features / (noise_features.amax(dim=1, keepdim=True) + 1e-6)

        dissimilarity = torch.abs(noise_features - noise_features.transpose(1, 2))
        dissimilarity.diagonal(dim1=1, dim2=2).fill_(-1)

        topk_indices = torch.topk(dissimilarity, k=k, dim=-1).indices
        attention_mask = torch.zeros(
            bsz, num_tokens, num_tokens, dtype=torch.bool, device=tokens.device
        )
        attention_mask.scatter_(2, topk_indices, True)
        attention_mask |= torch.eye(num_tokens, device=tokens.device, dtype=torch.bool).unsqueeze(0)

        normalized_tokens = F.normalize(tokens_fp32, dim=-1)
        attention_scores = torch.matmul(normalized_tokens, normalized_tokens.transpose(1, 2))
        attention_scores = attention_scores + noise_features.transpose(1, 2)
        attention_scores = attention_scores.masked_fill(
            ~attention_mask, torch.finfo(attention_scores.dtype).min
        )

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attended_tokens = torch.matmul(attention_probs, tokens_fp32)
        output = tokens_fp32 + noise_features * (attended_tokens - tokens_fp32)
        return output.to(dtype=tokens.dtype)

    # new
    def _apply_noise_guided_amplification(self, features, x):
        # NAA can run with or without noise_view. When noise_view is disabled,
        # it falls back to a standalone noise map generated by self.noise_extractor.
        if not self.use_noise_guided_amplification or not hasattr(self, "noise_extractor"):
            return features

        noise_map = self._last_noise_map
        if noise_map is None or noise_map.size(0) != x.size(0) or noise_map.shape[-2:] != x.shape[-2:]:
            noise_map = self.noise_extractor(x)

        if features.dim() == 3 and features.size(0) > 1:
            token_count = features.size(0) - 1
            token_side = int(token_count ** 0.5)
            if token_side * token_side != token_count:
                return features

            noise_features = F.adaptive_avg_pool2d(noise_map, (token_side, token_side))
            noise_features = noise_features.flatten(2).transpose(1, 2)
            image_tokens = features[1:].permute(1, 0, 2)
            image_tokens = self._noise_guided_topk_attention(image_tokens, noise_features)
            return torch.cat([features[:1], image_tokens.permute(1, 0, 2)], dim=0)

        if features.dim() == 4:
            batch_size, channels, height, width = features.shape
            target_height, target_width = height, width
            max_tokens = 1024
            if height * width > max_tokens:
                scale = (max_tokens / float(height * width)) ** 0.5
                target_height = max(1, min(height, int(round(height * scale))))
                target_width = max(1, min(width, int(round(width * scale))))

            pooled_features = F.adaptive_avg_pool2d(features, (target_height, target_width))
            noise_features = F.adaptive_avg_pool2d(noise_map, (target_height, target_width))

            image_tokens = pooled_features.flatten(2).transpose(1, 2)
            noise_features = noise_features.flatten(2).transpose(1, 2)
            image_tokens = self._noise_guided_topk_attention(image_tokens, noise_features)
            output = image_tokens.transpose(1, 2).view(batch_size, channels, target_height, target_width)

            if target_height != height or target_width != width:
                output = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)

            return output

        return features

    # ADD
    def set_label(self, label):
        self.label = label

    def feature_extraction(self, x, noise_view = False):
        # print(self.name)
        if self.name == "RN50" or self.name=="ViT-L/14" or self.name == "GeoRSCLIP":
            features = self.model.encode_image(x)
            if self.intermidiate_layer_output:
                features = features[self.intermidiate_layer_output]
            # choose the last layer
            else:
                if self.name == "RN50":
                    features = features["layer4"]
                else:
                    features = features["layer23"]

            # ADD
            if self.use_noise_view and not noise_view:
                # features = self.NoiseViewGenerator(features, x)
                # DCT频域处理
                dct_map = self.noise_extractor(x)  # [B, 1, H, W]
                self._last_noise_map = dct_map
                dct_input = dct_map.repeat(1, 3, 1, 1)  # 通道复制 [B, 3, H, W]
                dct_features = self.feature_extraction(dct_input, noise_view=True)  # [257, B, 1024]
                # 特征融合
                features = self.noise_fusion(features, dct_features) # [257, B, 1024]
                # print(f"Using noise view: {self.use_noise_view}, features shape: {features.shape}")

            # Apply NAA on encoder tokens right after noise fusion and before ASPP.
            if self.use_noise_guided_amplification and not noise_view:
                features = self._apply_noise_guided_amplification(features, x)

            # ADD
            if self.use_aspp:
                features = self.aspp_adapter(features)
            # ADD
            if self.use_conprn and self.label != None:
                self.con_loss = self.conprn(features, self.label) 
                # print(f"Contrastive loss: {self.con_loss.item()}")
            

        # ViT+RN fusion
        elif "RN50" in self.name and "ViT-L/14" in self.name:
            # given ViT feature layer
            features_vit = self.model[0].encode_image(x)[self.intermidiate_layer_output]
            features_vit = self._feature_map_transform(features_vit[1:])
            # explicit RN50 3rd layer to match the feature dimension
            features_rn50 = self.model[1].encode_image(x)["layer3"]
            features_rn50 = F.interpolate(features_rn50, size=(16, 16), mode='bilinear', align_corners=False)
            features = torch.cat([features_vit, features_rn50], 1)
        # for xceptionnet
        else:
            features = self.model(x)
            features = features[0]

        return features
                
    def forward(self, x):
        # Feature extraction
        self._last_noise_map = None
        features = self.feature_extraction(x)

        # ADD: SimDet
        if self.use_simdet:
            self.det_pred_probs = self.detector(features)
            # image_labels = self.detector.get_image_labels(self.label)
            # self.det_loss, self.det_metrics = self.detector.compute_loss_and_metrics(pred_probs, image_labels)
            # print(f"SimDet loss: {self.det_loss.item()}, metrics: {self.det_metrics}")
        
        # Forward step
        # ViT+RN fusion convolutional decoder
        if "RN50" in self.name and "ViT-L/14" in self.name and "conv" in self.decoder_type:
            output = self.fc(features)
        
        # Linear decoder
        elif self.decoder_type == "linear":
            # xceptionnet + linear
            if self.name == "xceptionnet":
                features = features.view(features.size()[0], features.size()[1], -1)
                features = features.permute(1, 0, 2)
                linear_outputs = [self.fc(input_part) for input_part in features[0:]]
            # CLIP + linear
            else:
                linear_outputs = [self.fc(input_part) for input_part in features[1:]]

            output = self._unify_linear_layer_outputs(linear_outputs)

        # Attention decoder
        elif self.decoder_type == "attention":
            features = self.att1(features)
            features = self.att2(features)
            linear_outputs = [self.fc(input_part) for input_part in features[1:]]
            output = self._unify_linear_layer_outputs(linear_outputs)

        # Convolutional decoder over RN
        elif "conv" in self.decoder_type and "RN50" == self.name:
            output = self.fc(features)
        
        # Convolutional decoder over ViT
        else:
            # print(features.shape)
            features = features[1:]
            # print(features.shape)
            output = self._feature_map_transform(features)
            output = self.fc(output)

        output = torch.flatten(output, start_dim =1)
        return output
