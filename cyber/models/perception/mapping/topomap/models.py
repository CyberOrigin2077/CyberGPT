from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import faiss
from opr.pipelines.registration import PointcloudRegistrationPipeline, RansacGlobalRegistrationPipeline, Feature2DGlobalRegistrationPipeline
from opr.models.place_recognition import MinkLoc3D


def get_place_recognition_model(config):
    WEIGHTS_PATH = config['weights_path']
    if config['model'] == 'minkloc3d':
        model = MinkLoc3D()
        model.load_state_dict(torch.load(WEIGHTS_PATH))
        model = model.to("cuda")
        model.eval()
        index = faiss.IndexFlatL2(256)
        return model, index
    elif config['model'] == 'mssplace':
        MODEL_CONFIG_PATH = config['model_config_path']
        model_config = OmegaConf.load(MODEL_CONFIG_PATH)
        model = instantiate(model_config)
        load = torch.load(WEIGHTS_PATH)
        model.load_state_dict(torch.load(WEIGHTS_PATH)['model_state_dict'])
        model = model.to("cuda")
        model.eval()
        index = faiss.IndexFlatL2(512)
        return model, index
    else:
        rospy.logfatal('Invalid place recognition model type {}. \
                        Parameter `model` for `place_recognition` must be `minkloc3d` or `mssplace`'.format(config['model']))
        return None, None


def get_registration_model(config):
    reg_model_type = config['model']
    if reg_model_type == 'feature2d':
        return Feature2DGlobalRegistrationPipeline(voxel_downsample_size=config['voxel_downsample_size'],
                                                   detector_type=config['detector_type'],
                                                   min_matches=config['min_matches'],
                                                   outlier_thresholds=config['outlier_thresholds'],
                                                   max_range=config['max_point_cloud_range'],
                                                   save_dir='/home/kirill/TopoSLAM/OpenPlaceRecognition/test_registration')
    elif reg_model_type == 'geotransformer':
        REGISTRATION_MODEL_CONFIG_PATH = config['model_config_path']
        REGISTRATION_WEIGHTS_PATH = config['weights_path']
        registration_model = instantiate(OmegaConf.load(REGISTRATION_MODEL_CONFIG_PATH))
        registration_model.load_state_dict(torch.load(REGISTRATION_WEIGHTS_PATH))
        return PointcloudRegistrationPipeline(model=registration_model,
                                              model_weights_path=REGISTRATION_WEIGHTS_PATH,
                                              device="cuda",  # the GeoTransformer currently only supports CUDA
                                              voxel_downsample_size=config['voxel_downsample_size'],  # recommended for geotransformer_kitti configuration
                                             )
    elif reg_model_type == 'icp':
        return RansacGlobalRegistrationPipeline(voxel_downsample_size=config['voxel_downsample_size'])
    else:
        print('Invalid registration model type {}. \
              Parameter `model` for `scan_matching` must be `feature2d`, `geotransformer`, or `icp`'.format(reg_model_type))
