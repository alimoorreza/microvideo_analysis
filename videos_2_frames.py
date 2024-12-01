import argparse
import cv2
import os


# REZA EDITS (11/28/24): modularized the code retaining 95% of what Sean and Nick wrote
def videos_2_frames(input_root, video_dir, output_root):
    
    # ------------------------------------------------------------------
    # frame extraction from videos
    # Sean and Nick Code Fusion
    # ------------------------------------------------------------------
    
    '''
    input_root  = "/nfs/jolteon/data/ssd/mdreza/tiktok_video_project/datasets/"
    output_root = "/nfs/jolteon/data/ssd/mdreza/tiktok_video_project/output/output_frames/"
    video_dir   = "sad_sample"    
    '''
    
    # REZA EDITS (11/28/24): batch processiong on a server
    source_dir  = input_root  + video_dir + '/'          #Folder where videos are located
    dest_dir    = output_root + video_dir + '/'
    
    #Set fps
    desired_fps = 1
    
    #If not created than make it
    os.makedirs(dest_dir, exist_ok=True)
    
    #Pull each video from folder to get frames
    #import pdb
    #pdb.set_trace()
    for filename in os.listdir(source_dir):
      #Set up cam to extract frames
      cam = cv2.VideoCapture(source_dir + filename)
      fps = cam.get(cv2.CAP_PROP_FPS)  # Get original video's FPS
    
      #Get name of video
      name, extension = os.path.splitext(filename)
    
      # Create folder to store current images from videos
      current_video_directory = dest_dir + name + "_frames/"
      os.makedirs(current_video_directory, exist_ok=True)
    
      currentframe = 0
      frames_saved = 0
      #Iterate over video and extract each frame and save to folder
      while(True):
        # reading from frame
        ret,frame = cam.read()
    
        if ret:
          if 1 + currentframe // (fps/desired_fps) > frames_saved:
              # Save the frame if it matches the desired interval
              name = current_video_directory + str(frames_saved).zfill(5) + '.png'
              print('Creating...' + name)
    
              # Writing the extracted images
              cv2.imwrite(name, frame)
              frames_saved += 1
        else:
          break
        currentframe += 1
      # Release all space and windows once done
      cam.release()
      cv2.destroyAllWindows()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process some parameters.")
    
    parser.add_argument("--input_root", type=str, default="/nfs/jolteon/data/ssd/mdreza/tiktok_video_project/datasets/", help="Directory where the input videos are located at")
    parser.add_argument("--video_dir", type=str, default="sad_videos", help="Name of specific type of videos 'sad_videos', 'happy_videos', or 'sad_sample' ")
    parser.add_argument("--output_root", type=str, default="/nfs/jolteon/data/ssd/mdreza/tiktok_video_project/output/output_frames/", help="Directory where the extracted frames will be saved.")
    
    args = parser.parse_args()
    
    videos_2_frames(input_root=args.input_root, video_dir=args.video_dir, output_root=args.output_root)




