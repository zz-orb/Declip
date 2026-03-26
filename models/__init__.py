from .clip_models import CLIPModelLocalisation


VALID_NAMES = [
    'CLIP:RN50',  
    'CLIP:ViT-L/14',
    'CLIP:xceptionnet',
    'CLIP:ViT-L/14,RN50', 
    'CLIP:GeoRSCLIP'
]

def get_model(opt):
    name, layer, decoder_type = opt.arch, opt.feature_layer, opt.decoder_type
    
    # ADD
    use_noise_view = opt.use_noise_view
    use_noise_guided_amplification = opt.use_noise_guided_amplification
    use_aspp = opt.use_aspp
    use_conprn = opt.use_conprn
    use_simdet = opt.use_simdet

    assert name in VALID_NAMES

    return CLIPModelLocalisation(name.split(':')[1], 
                                 intermidiate_layer_output = layer, 
                                 decoder_type=decoder_type,
                                 use_noise_view=use_noise_view,
                                 use_aspp=use_aspp,
                                 use_noise_guided_amplification=use_noise_guided_amplification,
                                 use_conprn=use_conprn,
                                 use_simdet=use_simdet) 
    
