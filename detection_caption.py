import argparse
import pickle
from argparse import Namespace
import gdown

import numpy as np
from PIL import Image as PIL_Image
import torch
import torchvision
from torchvision.models import detection
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torchvision.models._meta import _COCO_CATEGORIES

from models.End_ExpansionNet_v2 import End_ExpansionNet_v2
from utils.language_utils import convert_vector_idx2word

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument("--image", type=str, required=True,
        help="path to the input image")
    parser.add_argument("--detection_model", type=str,
        choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
        default="retinanet",
        help="name of the object detection model")
    parser.add_argument("--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = parser.parse_args()
    

    print("Downloading model weights ...")
    file_id = "1TbT1Bci49CavAk0sSJKWnoyHuB6qHYst"
    output = "./rf_model.pth"
    gdown.download(id=file_id, output=output, quiet=False)

    # Image captioning 

    model_dim = 512
    N_enc = 3
    N_dec = 3
    max_seq_len = 74
    load_path = './rf_model.pth'
    beam_size = 5
    img_size = 384

    drop_args = Namespace(enc=0.0,
                          dec=0.0,
                          enc_input=0.0,
                          dec_input=0.0,
                          other=0.0)
    model_args = Namespace(model_dim=model_dim,
                           N_enc=N_enc,
                           N_dec=N_dec,
                           dropout=0.0,
                           drop_args=drop_args)

    print("Captioning task running...")
    with open('./demo_coco_tokens.pickle', 'rb') as f:
        coco_tokens = pickle.load(f)
    print("Dictionary loaded ...")

    model = End_ExpansionNet_v2(swin_img_size=img_size, swin_patch_size=4, swin_in_chans=3,
                                swin_embed_dim=192, swin_depths=[2, 2, 18, 2], swin_num_heads=[6, 12, 24, 48],
                                swin_window_size=12, swin_mlp_ratio=4., swin_qkv_bias=True, swin_qk_scale=None,
                                swin_drop_rate=0.0, swin_attn_drop_rate=0.0, swin_drop_path_rate=0.0,
                                swin_norm_layer=torch.nn.LayerNorm, swin_ape=False, swin_patch_norm=True,
                                swin_use_checkpoint=False,
                                final_swin_dim=1536,
                                d_model=model_args.model_dim, N_enc=model_args.N_enc,
                                N_dec=model_args.N_dec, num_heads=8, ff=2048,
                                num_exp_enc_list=[32, 64, 128, 256, 512],
                                num_exp_dec=16,
                                output_word2idx=coco_tokens['word2idx_dict'],
                                output_idx2word=coco_tokens['idx2word_list'],
                                max_seq_len=max_seq_len, drop_args=model_args.drop_args,
                                rank='cpu')
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded ...")

    transf_1 = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size))])
    transf_2 = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])

    path = args.image
    pil_image = PIL_Image.open(path)
    if pil_image.mode != 'RGB':
        pil_image = PIL_Image.new("RGB", pil_image.size)
    preprocess_pil_image = transf_1(pil_image)
    tens_image_1 = torchvision.transforms.ToTensor()(preprocess_pil_image)
    tens_image_2 = transf_2(tens_image_1)
    input_image = tens_image_2

    print("Generating captions ...\n")
    image = input_image.unsqueeze(0)
    beam_search_kwargs = {'beam_size': beam_size,
                            'beam_max_seq_len': max_seq_len,
                            'sample_or_max': 'max',
                            'how_many_outputs': 1,
                            'sos_idx': coco_tokens['word2idx_dict'][coco_tokens['sos_str']],
                            'eos_idx': coco_tokens['word2idx_dict'][coco_tokens['eos_str']]}
    with torch.no_grad():
        pred, _ = model(enc_x=image,
                        enc_x_num_pads=[0],
                        mode='beam_search', **beam_search_kwargs)
    pred = convert_vector_idx2word(pred[0][0], coco_tokens['idx2word_list'])[1:-1]
    pred[-1] = pred[-1] + '.'
    pred = ' '.join(pred).capitalize()
    caption = pred
    

    # Object detection
    print("Object detection task running...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CLASSES = _COCO_CATEGORIES
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    MODELS = {
        "frcnn-resnet": detection.fasterrcnn_resnet50_fpn_v2,
        "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_fpn,
        "retinanet": detection.retinanet_resnet50_fpn_v2
    }
    WEIGHTS = {
        "frcnn-resnet": detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1,
        "frcnn-mobilenet": detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1,
        "retinanet": detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    }

    weights = WEIGHTS[args.detection_model]
    model = MODELS[args.detection_model](weights=weights, progress=True, num_classes=len(CLASSES)).to(DEVICE)
    model.eval()
    print("Model loaded ...")

    img = read_image(args.image).to(DEVICE)

    preprocess = weights.transforms()

    batch = [preprocess(img)]

    prediction = model(batch)[0]
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box_list = prediction['boxes'].detach().cpu().numpy().tolist()

    print(f"\n\n\n{'='*30}\nRESULT\n{'='*30}\n")
    print(f"\nImage: {path}")
    print("Generated caption: {caption}")
    print("Detected objects: ")

    for i in range(len(box_list)):
        confidence = prediction["scores"][i]
        if confidence > args.confidence:
            box = f"[{box_list[i][0]:.2f}, {box_list[i][1]:.2f}, {box_list[i][2]:.2f}, {box_list[i][3]:.2f}]"
            print(f"label: {labels[i]}, bbox: {box}, confidence: {confidence:.3f}")
    box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4)
    im = to_pil_image(box.detach())
    im.show()

    print("="*30)
