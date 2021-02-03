from predictor import Predictor
from multi_modal_wrapper import MultiModalWrapper
from dataset_manager import DatasetManager
import argparse
from collections import OrderedDict
import torch
from PIL import Image
import os

test_config_file = 'C:/Users/myyu/source/CASIA-SURF_CeFA/rgb_track/experiment_tests/protocol_4_1/protocol_4_1.config'
checkpoint_file = 'C:/Users/myyu/source/CASIA-SURF_CeFA/rgb_track/experiments/rgb_track/exp1_protocol_4_1/checkpoints/model_0.pth'
model_file = 'C:/Users/myyu/source/CASIA-SURF_CeFA/rgb_track/experiments/rgb_track/exp1_protocol_4_1/rgb_track_exp1_protocol_4_1.config'

def rgb_loader(path):
    return Image.open(path).convert('RGB')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--file_path',
                        type=str,
                        help='Path to checkpoint')  
    args = parser.parse_args()

    test_config = torch.load(test_config_file) 
    model_config = torch.load(model_file) 
    test_config.out_path = os.path.join('experiment_tests/', test_config.test_config_name)

    test_config.dataset_configs.file_path = args.file_path
    model_config.wrapper_config.no_batch = True
    
    # change out_path in test_config
    out_path = os.path.join(test_config.out_path,
                            model_config.head_config.task_name,
                            model_config.head_config.exp_name)
    os.makedirs(out_path, exist_ok=True)
    test_config.out_path = out_path

    predictor = Predictor(test_config, model_config, checkpoint_file)
    item_dict = OrderedDict()
    item_dict['data'] = [os.path.join(args.file_path, x) for x in sorted(os.listdir(args.file_path))]
    img = [rgb_loader(x) for x in item_dict['data']]
    output = predictor.run_predict_one(img)
    print(round(output, 4))

