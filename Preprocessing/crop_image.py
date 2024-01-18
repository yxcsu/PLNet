from PIL import Image
  
def crop_image(img):
    # Use Image to read images and convert them to RGB images
    # img = Image.open(image).convert('RGB')
    # Obtain the width and height of the image
    width, height = img.size
    # Create an empty list to store cropped images
    blocks = []
    # Starting from the top left corner of the image, traverse each row and column of the image
    # for x in range(width):
    #     x1 = img.crop((x,0,x+1,height))
    #     if x1.getextrema() != ((0, 0), (0, 0), (0, 0)):
    #         w=x
    #         # print(h)
    #         break
    # for y in range(height):
    #     x2 = img.crop((0,y,width,y+1))
    #     if x2.getextrema() != ((0, 0), (0, 0), (0, 0)):                
    #         h=y
    #         break
    # for x in range(w, width, 224):
    #     for y in range(h, height, 224):
    for x in range(0, width, 224):
        for y in range(0, height, 224):
            block = img.crop((x,y,x+224,y+224))
            blocks.append(block)
    # Finally, output this list
    return blocks