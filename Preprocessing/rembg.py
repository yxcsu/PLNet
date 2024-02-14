from io import BytesIO
import os
from PIL import Image
from rembg import remove
from tqdm import tqdm
def process_images(input_path, output_path):
    '''
    Remove the background image of the corresponding path and save it
    Args:
        input_path (string): The path of the input images
        output_path (string): The path of the output images
    '''

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for image_name in tqdm(os.listdir(input_path)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_image_path = os.path.join(input_path, image_name)
            input_image = Image.open(full_image_path)
            # output_image_data = remove(input_image)
            with open(full_image_path,"rb" ) as f:
                t = f.read()
            result_a = remove(data=t)
            output_image_io = BytesIO(result_a)
            output_image = Image.open(output_image_io).convert('RGB')
            bbox = output_image.getbbox()
            cropped_original = input_image.crop(bbox)
            # cropped_original.save(os.path.join(output_path, image_name))
            if cropped_original.mode == 'RGBA':
                cropped_original = cropped_original.convert('RGB')
            # Save the cropped part of the original image to the output path
            cropped_original.save(os.path.join(output_path, image_name.replace('.png', '.jpg')), 'JPEG')
if __name__ == '__main__':
    input_path, output_path = 'path to load the images', 'path to save the images'
    process_images(input_path, output_path)
