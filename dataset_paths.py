def get_dolos_localisation_dataset_paths(dataset):
    paths = dict(
        fake_path=f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        masks_path=f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        key=dataset
    )
    return paths

def get_dolos_detection_dataset_paths(dataset):
    paths = dict(
        real_path=f'datasets/dolos_data/celebahq/fake/{dataset}/images/test',
        fake_path=f'datasets/dolos_data/celebahq/real/{dataset}/test',
        masks_path=f'datasets/dolos_data/celebahq/fake/{dataset}/masks/test',
        key=dataset
    ),
    return paths

def get_autosplice_localisation_dataset_paths(compression):
    paths = dict(
        fake_path=f'datasets/AutoSplice/Forged_JPEG{compression}',
        masks_path=f'datasets/AutoSplice/Mask',
        key=f'autosplice_jpeg{compression}'
    )
    return paths

def get_DOTA_PS_sample_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/nas/datasets/DOTA-PS',
        fake_path=f'data-ps-sample.txt',
        key=dataset
    )
    return paths

def get_DOTA_PS_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/zhaoxiaohan_2021/datasets/DOTA-PS',
        fake_path=f'data-ps.txt',
        key=dataset
    )
    return paths

def get_DOTA_PS_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/zhaoxiaohan_2021/datasets/DOTA-PS',
        fake_path=f'data-ps-all.txt',
        key=dataset
    )
    return paths

def get_DOTA_Deepfake_localisation_dataset_paths(dataset):
    paths = dict(
        # root_path ='/irip/zhaoxiaohan_2021/datasets/DOTA-deepfake',
        # fake_path=f'DOTA-deepfake-type1all.txt',
        root_path ='/nas/datasets/DOTA-deepfake',
        fake_path=f'DOTA-deepfake-all.txt',
        key=dataset
    )
    return paths

def get_Both_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/zhaoxiaohan_2021/datasets/RS-Data/',
        fake_path=f'test_both.txt',
        key=dataset
    )
    return paths

def get_SIOR_simdet_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/zhaoxiaohan_2021/datasets/RS-Data/SIOR/fake_images',
        fake_path=f'SIOR_obj_simdet_test.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur3_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gaussianblur3.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur5_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gaussianblur5.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur15_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gaussianblur15.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise3_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gaussiannoise3.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise5_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gaussiannoise5.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise15_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gaussiannoise15.txt',
        key=dataset
    )
    return paths

def get_Noise_jpeg90_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'jpeg90.txt',
        key=dataset
    )
    return paths

def get_Noise_jpeg85_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'jpeg85.txt',
        key=dataset
    )
    return paths

def get_Noise_resize75_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'resize75.txt',
        key=dataset
    )
    return paths

def get_Noise_resize50_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'resize50.txt',
        key=dataset
    )
    return paths

def get_Noise_resize25_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'resize25.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur3_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gb3_all.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur5_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gb5_all.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur15_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gb15_all.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise3_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gn3_all.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise5_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gn5_all.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise15_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'gn15_all.txt',
        key=dataset
    )
    return paths

def get_Noise_jpeg90_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'jpeg90_all.txt',
        key=dataset
    )
    return paths

def get_Noise_jpeg85_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'jpeg85_all.txt',
        key=dataset
    )
    return paths

def get_Noise_resize75_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'resize75_all.txt',
        key=dataset
    )
    return paths

def get_Noise_resize50_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'resize50_all.txt',
        key=dataset
    )
    return paths

def get_Noise_resize25_all_localisation_dataset_paths(dataset):
    paths = dict(
        root_path ='/irip/suoyucong_2020/datasets/DOTA-PS',
        fake_path=f'resize25_all.txt',
        key=dataset
    )
    return paths

def get_DOTA_PS_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'input.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur3_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'gaussianblur3.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur5_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'gaussianblur5.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussianblur15_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'gaussianblur15.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise3_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'gaussiannoise3.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise5_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'gaussiannoise5.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_gaussiannoise15_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'gaussiannoise15.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_jpeg90_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'jpeg90.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_jpeg85_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'jpeg85.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_resize75_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'resize75.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_resize50_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'resize50.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

def get_Noise_resize25_detection_dataset_paths(dataset):
    paths = dict(
        root_path=f'/irip/suoyucong_2020/datasets/DOTA-PS/',
        fake_path=f'resize25.txt',
        real_path=f'real.txt',
        key=dataset
    )
    return paths

LOCALISATION_DATASET_PATHS = [
    # get_SIOR_simdet_localisation_dataset_paths('SIOR_obj_simdet'),

    get_DOTA_PS_sample_localisation_dataset_paths('DOTA-PS-sample'),
    
    # get_DOTA_Deepfake_localisation_dataset_paths('DOTA-deepfake'),
    # get_DOTA_PS_all_localisation_dataset_paths('DOTA-PS-all'),

    # get_DOTA_PS_sample_localisation_dataset_paths('DOTA-PS-sample'),
    # get_Both_localisation_dataset_paths('Both'),

    # get_Noise_gaussianblur3_all_localisation_dataset_paths('Noise-gaussianblur3-all'),
    # get_Noise_gaussianblur5_all_localisation_dataset_paths('Noise-gaussianblur5-all'),
    # get_Noise_gaussianblur15_all_localisation_dataset_paths('Noise-gaussianblur15-all'),
    # get_Noise_gaussiannoise3_all_localisation_dataset_paths('Noise-gaussiannoise3-all'),
    # get_Noise_gaussiannoise5_all_localisation_dataset_paths('Noise-gaussiannoise5-all'),
    # get_Noise_gaussiannoise15_all_localisation_dataset_paths('Noise-gaussiannoise15-all'),
    # get_Noise_jpeg90_all_localisation_dataset_paths('Noise-jpeg90-all'),
    # get_Noise_jpeg85_all_localisation_dataset_paths('Noise-jpeg85-all'),
    # get_Noise_resize75_all_localisation_dataset_paths('Noise-resize75-all'),
    # get_Noise_resize50_all_localisation_dataset_paths('Noise-resize50-all'),
    # get_Noise_resize25_all_localisation_dataset_paths('Noise-resize25-all'), 

    
    # get_Noise_gaussianblur3_localisation_dataset_paths('Noise-gaussianblur3'),
    # get_Noise_gaussianblur5_localisation_dataset_paths('Noise-gaussianblur5'),
    # get_Noise_gaussianblur15_localisation_dataset_paths('Noise-gaussianblur15'),
    # get_Noise_gaussiannoise3_localisation_dataset_paths('Noise-gaussiannoise3'),
    # get_Noise_gaussiannoise5_localisation_dataset_paths('Noise-gaussiannoise5'),
    # get_Noise_gaussiannoise15_localisation_dataset_paths('Noise-gaussiannoise15'),
    # get_Noise_jpeg90_localisation_dataset_paths('Noise-jpeg90'),
    # get_Noise_jpeg85_localisation_dataset_paths('Noise-jpeg85'),
    # get_Noise_resize75_localisation_dataset_paths('Noise-resize75'),
    # get_Noise_resize50_localisation_dataset_paths('Noise-resize50'),
    # get_Noise_resize25_localisation_dataset_paths('Noise-resize25'), 

    # get_dolos_localisation_dataset_paths('pluralistic'),
    # get_dolos_localisation_dataset_paths('lama'),
    # get_dolos_localisation_dataset_paths('repaint-p2-9k'),
    # get_dolos_localisation_dataset_paths('ldm'),

    # get_autosplice_localisation_dataset_paths("75"),
    # get_autosplice_localisation_dataset_paths("90"),
    # get_autosplice_localisation_dataset_paths("100"),
]


DETECTION_DATASET_PATHS = [
    get_DOTA_PS_detection_dataset_paths('DOTA-PS-det'),
    get_Noise_gaussianblur3_detection_dataset_paths('Noise-gaussianblur3-det'),
    get_Noise_gaussianblur5_detection_dataset_paths('Noise-gaussianblur5-det'),
    get_Noise_gaussianblur15_detection_dataset_paths('Noise-gaussianblur15-det'),
    get_Noise_gaussiannoise3_detection_dataset_paths('Noise-gaussiannoise3-det'),
    get_Noise_gaussiannoise5_detection_dataset_paths('Noise-gaussiannoise5-det'),
    get_Noise_gaussiannoise15_detection_dataset_paths('Noise-gaussiannoise15-det'),
    get_Noise_jpeg90_detection_dataset_paths('Noise-jpeg90-det'),
    get_Noise_jpeg85_detection_dataset_paths('Noise-jpeg85-det'),
    get_Noise_resize75_detection_dataset_paths('Noise-resize75-det'),
    get_Noise_resize50_detection_dataset_paths('Noise-resize50-det'),
    get_Noise_resize25_detection_dataset_paths('Noise-resize25-det'),

    # get_Noise_gaussianblur3_detection_dataset_paths('Noise-gaussianblur3'),
    # get_dolos_detection_dataset_paths('pluralistic'),
    # get_dolos_detection_dataset_paths('lama'),
    # get_dolos_detection_dataset_paths('repaint-p2-9k'),
    # get_dolos_detection_dataset_paths('ldm'),
    # TO BE PUBLISHED
    # get_dolos_detection_dataset_paths('ldm_clean'),
    # get_dolos_detection_dataset_paths('ldm_real'),
]
