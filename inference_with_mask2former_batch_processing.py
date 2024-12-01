# Reza, Sean and Nick Code Fusion

import argparse

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="mask2former")

# import some common libraries
import numpy as np
import cv2
from PIL import Image
import os
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")

# import Mask2Former project
from mask2former import add_maskformer2_config
IM_WIDTH  = 640
IM_HEIGHT = 480

print("Is CUDA available?", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

def init_mask2former():
    
    cfg = get_cfg()
    print(f"{cfg.MODEL.DEVICE}")
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    predictor = DefaultPredictor(cfg)

    return predictor
        
    #----------------------------------


def frames_to_sem_segmentation(input_root, video_dir, output_root):
    
    # ------------------------------------------------------------------
    # frame extraction from videos
    # Sean and Nick Code Fusion
    # ------------------------------------------------------------------

    predictor = init_mask2former()

    
    '''
    root = "/content/drive/MyDrive/Fall2024Research/TikTok/"    
    source_dir  = root + 'TestVideos/'  #Folder where videos are located
    dest_dir    = root + 'FrameSplits/'
    '''
    
    '''
    input_root  = "/nfs/jolteon/data/ssd/mdreza/tiktok_video_project/output/output_frames/"
    output_root = "/nfs/jolteon/data/ssd/mdreza/tiktok_video_project/output/output_semsegs/"
    video_dir   = "sad_sample"    
    '''
   
    
    # REZA EDITS (11/28/24): batch processiong on a server
    source_dir  = input_root  + video_dir + '/'          #Folder where videos are located
    dest_dir    = output_root + video_dir + '/'

    
    os.makedirs(dest_dir, exist_ok=True)
    
    image_dest  = dest_dir + "/images"
    masks_dest  = dest_dir + "/masks"
    labels_dest = dest_dir + "/labels"
    
    os.makedirs(image_dest, exist_ok=True)
    os.makedirs(masks_dest, exist_ok=True)
    os.makedirs(labels_dest, exist_ok=True)
    
    # Go through all of the RGB frames generated from above
    
    for foldername in os.listdir(source_dir):
    
        current_frame = 0
        print("Starting On: " + foldername)
    
        curr_images_dest  = image_dest  + "/" + foldername + "/"
        curr_masks_dest   = masks_dest  + "/" + foldername + "/"
        curr_labels_dest  = labels_dest + "/" + foldername + "/"
    
        os.makedirs(curr_images_dest, exist_ok=True)
        os.makedirs(curr_masks_dest, exist_ok=True)
        os.makedirs(curr_labels_dest, exist_ok=True)
    
        # Prepare a list to collect labels
        labels_list = []
    
        for image in os.listdir(source_dir + foldername):
            # Image to segment
            im = cv2.imread(source_dir + foldername + "/" + image)
    
            # Output
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
            sem_seg = outputs["sem_seg"].argmax(0).to("cpu")
    
    
            #Save image
            cv2.imwrite(curr_images_dest + str(current_frame).zfill(5) + ".png", im)
    
            #Save mask
            segmented_mask = outputs["sem_seg"].argmax(0).to("cpu").numpy().astype(np.uint8)
    
            # Initialize a color mask with the same dimensions as the original mask
            color_mask = np.zeros((segmented_mask.shape[0], segmented_mask.shape[1], 3), dtype=np.uint8)
    
            # Map each label in the segmented mask to its corresponding color
            for label in range(len(v.metadata.stuff_classes)):
                # Create a mask for the current label
                mask = (segmented_mask == label)
                # Get the corresponding color
                if label < len(v.metadata.stuff_colors):
                    color = v.metadata.stuff_colors[label]
                    # Apply the color to the color_mask where the current label is found
                    color_mask[mask] = [color[2], color[1], color[0]]  # color is expected to be in the format[RGB]
    
            cv2.imwrite(curr_masks_dest + str(current_frame).zfill(5) + ".png", color_mask)
    
            # Save labels
            unique_labels, areas = np.unique(sem_seg, return_counts=True)
    
    
            # Prepare a list to collect labels for the current frame
            frame_labels_list = []
    
            # Collect labels in the desired format
            for label in filter(lambda l: l < len(v.metadata.stuff_classes), unique_labels):
                # Get the RGB color for the current label
                color = v.metadata.stuff_colors[label]
                # Format the string
                label_entry = f"({color[0]}, {color[1]}, {color[2]}) = {v.metadata.stuff_classes[label]}"
                frame_labels_list.append(label_entry)  # Append to the frame_labels_list
    
            # Save the collected labels to a text file
            with open(curr_labels_dest + f"{current_frame}".zfill(5) + ".txt", "w") as label_file:
                for frame_labels in frame_labels_list:
                    label_file.write(frame_labels + "\n")
    
    
            current_frame += 1
            print(f"Finished Saves for frame {current_frame}")

    
# ------------------------------------------------------------------



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument("--input_root", type=str, default="/nfs/jolteon/data/ssd/mdreza/tiktok_video_project/output/output_frames/", help="Directory where the video frames are located at")
    parser.add_argument("--video_dir", type=str, default="sad_sample", help="Name of specific type of videos 'sad_videos', 'happy_videos', or 'sad_sample' ")
    parser.add_argument("--output_root", type=str, default="/nfs/jolteon/data/ssd/mdreza/tiktok_video_project/output/output_semsegs/", help="Directory where the semantic segmentation results will be saved.")
    
    args = parser.parse_args()
    
    frames_to_sem_segmentation(input_root=args.input_root, video_dir=args.video_dir, output_root=args.output_root)











    


'''
#----------------------------------
def inference_using_mask2former(input_img_name, sv_path, sv_name):
    cfg = get_cfg()
    print(f"{cfg.MODEL.DEVICE}")
    
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl'
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
    predictor = DefaultPredictor(cfg)
    
    im      = cv2.imread(input_img_name) # cv2.imread("./nyud_v2_rgb.png")
    im      = cv2.resize(im, (IM_WIDTH, IM_HEIGHT))
    outputs = predictor(im)
    
    # Show "Panoptic segmentation (top), instance segmentation (middle), semantic segmentation (bottom)"
    v = Visualizer(im[:, :, ::-1], coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    sem_seg = outputs["sem_seg"].argmax(0).to("cpu")
    semantic_result = v.draw_sem_seg(sem_seg).get_image()
    #cv2_imshow(semantic_result)
    
    #save_img = Image.fromarray(color_img.astype(np.uint8))
    save_img = Image.fromarray(semantic_result)
    save_img.save(os.path.join(sv_path, sv_name))
    
    #----------------------------------
'''

    


