#%%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

%matplotlib qt
def generate_yolo_labels_from_numpy(image_name, image_array, columns, output_dir):
    """
    Generate YOLO format labels from atomic column data for a given NumPy image array.

    Parameters:
    - image_name: Name of the image (without extension).
    - image_array: The NumPy array representing the image.
    - columns: A list of tuples, each containing (x_center, y_center, class_id).
               x_center and y_center should be in pixel coordinates.
    - output_dir: Directory where the label files will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_height, image_width = image_array.shape[:2]

    label_path = os.path.join(output_dir, f"{image_name}.txt")
    
    with open(label_path, 'w') as f:
        for column in columns:
            x_center, y_center, sigma, class_id = column
            
            # Normalize coordinates to [0, 1] range
            x_center_norm = x_center / image_width
            y_center_norm = y_center / image_height
            
            # YOLO format requires x_center, y_center, width, height
            bbox_width_norm = 2*sigma / image_width  # Adjust based on your specific use case
            bbox_height_norm = 2*sigma / image_height  # Same as above
            
            # Write to the label file in YOLO format
            f.write(f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {bbox_width_norm:.6f} {bbox_height_norm:.6f}\n")

    print(f"Label saved to {label_path} for image {image_name}")

def save_image(image_name, image_array, output_dir):
    """
    Save the given NumPy image array to the specified output directory.

    Parameters:
    - image_name: Name of the image (without extension).
    - image_array: The NumPy array representing the image.
    - output_dir: Directory where the image file will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_path = os.path.join(output_dir, f"{image_name}.png")
    # convert image_array to uint8
    image_array = (image_array - np.min(image_array))/(np.max(image_array)-np.min(image_array))*255
    cv2.imwrite(image_path, image_array)

    print(f"Image saved to {image_path}")
    plt.figure()
    plt.imshow(image_array,cmap='gray')
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    x, y, s, c = zip(*columns)
    x= np.array(x)
    y= np.array(y)
    c = np.array(c).astype(int)
    c = np.clip(c, 0, len(colors)-1)  # Ensure c is within the range of valid indices
    plt.scatter(x, y, c=[colors[i] for i in c], marker='x')  # Use list comprehension to map indices to colors

    annotated_folder = os.path.join(output_dir, f'../annotated')
    if not os.path.exists(annotated_folder):
        os.makedirs(annotated_folder, exist_ok=True)
    annotated_file = os.path.join(annotated_folder, f"{image_name}.png")
    plt.savefig(annotated_file, dpi=300)
    plt.close()
#%%
from PIL import Image
import random
from qem.image_fitting import ImageModelFitting

def select_random_region(image_array, region_size):
    """
    Selects a random region from an image.

    Parameters:
    - image_path: str, path to the image file.
    - region_size: tuple, (height, width) of the region to extract.

    Returns:
    - region: PIL Image, the extracted random region.
    """
    
    # Load the image and convert it to a NumPy array


    # Get the dimensions of the image
    image_height, image_width = image_array.shape

    # Calculate the maximum top-left corner (y, x) that can be chosen
    max_x = image_width - region_size[1]
    max_y = image_height - region_size[0]

    # Randomly select the top-left corner within the allowable range
    random_x = random.randint(0, max_x)
    random_y = random.randint(0, max_y)

    # Extract the region
    region_np = image_array[random_y:random_y + region_size[0], random_x:random_x + region_size[1]]

    # Convert the numpy array back to an image
    return region_np
# %%
mode = 'train'  # 'train' or 'val'
output_dir = f"dataset/{mode}/labels"  # Directory where the label files will be saved

# Example data: image names as keys and NumPy arrays as values
images_data = dict()

# Corresponding atomic column data for each image
columns_data = dict()

file = '/home/zzhang/OneDrive/code/qem/data/STO/adf_average_STO.txt'
image = np.loadtxt(file)

for i in range(12,13):
    image_name = f"image_{i}"
    # select random part of the image with a size of 256 x 256
    image_region = select_random_region(image, (512, 512))
    dx= 0.1645429228960236	
    model=ImageModelFitting(image_region, dx=dx,units='A')
    model.find_peaks()
    cif_file = '/home/zzhang/OneDrive/code/qem/data/STO/SrTiO3_mp-5229_conventional_standard.cif'
    model.map_lattice(cif_file=cif_file,elements=['Sr','Ti'],min_distance=10, reciprocal=True)
    # model.select_region()
    # model.fit_background = True
    # params = model.init_params()
    # model.fit_global(params)
    # model.coordinates = np.array([model.params['pos_x'],model.params['pos_y']]).T
    model.add_or_remove_peaks(min_distance=10)
    
    sigma =5
    images_data[image_name] = model.image
    columns_data[image_name] = [(x, y, s, c) for x, y, s, c in zip(model.coordinates[:,0],model.coordinates[:,1], np.tile(sigma,len(model.atom_types)),model.atom_types)]

# Generate labels for each image
for image_name, image_array in images_data.items():
    columns = columns_data.get(image_name, [])
    # save image in data/images/train
    save_image(image_name, image_array, f'dataset/{mode}/images')
    generate_yolo_labels_from_numpy(image_name, image_array, columns, output_dir)
#%%
import numpy as np
import matplotlib.pyplot as plt
import PIL 

file = '/home/zzhang/OneDrive/code/qem/data/STO/adf_average_STO.txt'
image = np.loadtxt(file)
# save the array to a png file
# plt.figure(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)
# plt.imshow(image, cmap='gray')
# plt.axis('off')
# plt.savefig('adf_average_STO.png', bbox_inches='tight', pad_inches=0, dpi=100)

# %%
from PIL import Image
import random
from qem.image_fitting import ImageModelFitting

def select_random_region(image_array, region_size):
    """
    Selects a random region from an image.

    Parameters:
    - image_path: str, path to the image file.
    - region_size: tuple, (height, width) of the region to extract.

    Returns:
    - region: PIL Image, the extracted random region.
    """
    
    # Load the image and convert it to a NumPy array


    # Get the dimensions of the image
    image_height, image_width = image_array.shape

    # Calculate the maximum top-left corner (y, x) that can be chosen
    max_x = image_width - region_size[1]
    max_y = image_height - region_size[0]

    # Randomly select the top-left corner within the allowable range
    random_x = random.randint(0, max_x)
    random_y = random.randint(0, max_y)

    # Extract the region
    region_np = image_array[random_y:random_y + region_size[0], random_x:random_x + region_size[1]]

    # Convert the numpy array back to an image
    return region_np

for i in range(0,1):
    region = select_random_region(image,(512,512))
    plt.figure(figsize=(region.shape[1] / 100, region.shape[0] / 100), dpi=100)
    plt.imshow(region, cmap='gray')
    plt.axis('off')
    plt.savefig(f'random_region_{i}.png', bbox_inches='tight', pad_inches=0, dpi=100)
# %%
from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n_atom.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="/home/zzhang/OneDrive/code/qem/examples/coco8.yaml", epochs=10, imgsz=640)
model.save("yolov8n_atom.pt")  # save the model to disk
# %%
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n_atom.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model(["random_region_0.png"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
# %%
