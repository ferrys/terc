import cv2
import os
import glob
##################################
## Resizes images to 224 x 224  ##
##################################

def resize_images(data_path, image_size=224):
    processed_dir = data_path + "/resized_images"
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    files = glob.glob(data_path + '/*.jpg')
    processed_files = glob.glob(processed_dir + '/*.jpg')

    files = glob.glob(data_path + '/*.jpg')
    processed_files = glob.glob(processed_dir + '/*.jpg')

    for file in files:
      image = cv2.imread(file)
      image = cv2.resize(image, (image_size, image_size),0,0)
      if os.name == 'nt':
        cv2.imwrite(processed_dir + '/' + file[file.rindex("\\")+1:], image)
      elif os.name == 'posix':
        cv2.imwrite(processed_dir + '/' + file[file.rindex("/")+1:], image)
      else:
        print("Unsupported operating system. Please use Windows or Mac.")
      


if __name__ == "__main__":
    resize_images('Terc_Images')