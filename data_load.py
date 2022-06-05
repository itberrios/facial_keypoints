import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2


class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])
        
        # image = mpimg.imread(image_name) # this normalizes image to 0-1
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # if image has an alpha color channel, get rid of it
        if(image.shape[2] == 4):
            image = image[:,:,0:3]
        
        key_pts = self.key_pts_frame.iloc[idx, 1:].values
        key_pts = key_pts.astype('float').reshape(-1, 2)
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample
    

class FerEmotionsDataset(Dataset):

    def __init__(self, csv_file, usage, pad=None, transform=None):
        ''' Inputs:
                csv_file - abspath to csv for FER data
                usage (str) - denotes train, valid, or test 
                            ('Training', 'PublicTest', 'PrivateTest')
                pad - number of pixel to pad images on all sides
                transform - transform object to be applied on the data
                '''
        # read entire DataFrame
        fer_df = pd.read_csv(csv_file)

        # get appropriate portion of the dataframe
        self.fer_df = fer_df[fer_df['Usage'] == usage]
        self.pad = pad
        self.transform = transform

    def __len__(self):
        return len(self.fer_df)

    def __getitem__(self, idx):
        # get image pixels
        pixel_str = self.fer_df.iloc[idx, 1]
        pixels = list(map(int, pixel_str.split(' ')))
        pixels = np.array(pixels).reshape((48, 48)).astype(np.float32)

        if self.pad:
            pixels_copy = pixels.copy()
            pixels = np.tile(pixels.mean(), (2*self.pad + 48, 2*self.pad + 48))
            pixels[self.pad:48 + self.pad, self.pad:48 + self.pad] = pixels_copy
            pixels = pixels.astype(np.float32)

        # convert to 3 channel grayscale image
        pixels = np.array([pixels, pixels, pixels]).transpose(1, 2, 0).astype(np.float32)
        # print(pixels.shape)

        # normalize to 0-1
        pixels = pixels / 255.0

        if self.transform:
            # pixels = self.transform(pixels)
            # manually transform data
            pixels = cv2.resize(pixels, (224, 224))

            # normalize
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            pixels = (pixels - mean) / std


            pixels = torch.tensor(pixels.transpose(2, 0, 1), dtype=torch.float)
            
            

        emotion = self.fer_df.iloc[idx, 0]

        return pixels, emotion

            



# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1].""" 

    def __init__(self, grayscale=False, mode=1):
        ''' grayscale determines whether to use gray scale or not
            mode (1 or 2) determines how to normalize the key points
            '''
        self.grayscale = grayscale     
        self.mode = mode

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale if applicable
        if self.grayscale:
            image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # get 3 layer grayscal image for transfer learning input
            image_copy = np.array([image_copy, 
                                   image_copy, 
                                   image_copy]).transpose(1, 2, 0)
        
        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy/255.0

        # standardize rgb image to 0 mean, 1 stdev
        if not self.grayscale:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image_copy = (image_copy - mean) / std
            
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        if self.mode == 1:
            key_pts_copy = (key_pts_copy - 100)/50.0
        elif self.mode == 2:
            pass # do nothing

        return {'image': image_copy, 'keypoints': key_pts_copy}


class AddColorJitter(object):
    """Convert a color image to grayscale and normalize the color range to [0,1].""" 

    def __init__(self, brightness, contrast, saturation, hue):
        ''' Applies brightness, constrast, saturation, and hue jitter to image.
            The inputs are tuples that contain min and max possbile adjustment ranges
            '''
        self.min_brightness = brightness[0]
        self.max_brightness = brightness[1]

        self.min_contrast = contrast[0]
        self.max_contrast = contrast[1]

        self.min_saturation = saturation[0]
        self.max_saturation = saturation[1]

        self.min_hue = hue[0]
        self.max_hue = hue[1]

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        
        # get image and key point copies
        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # get random parameter values
        brightness = np.random.uniform(self.min_brightness, self.max_brightness)
        contrast = np.random.uniform(self.min_contrast, self.max_contrast)
        saturation = np.random.uniform(self.min_saturation, self.max_saturation)
        hue = np.random.uniform(self.min_hue, self.max_hue)

        # convert image copy to PIL image
        image_copy = Image.fromarray(image_copy)

        # apply color jitter to image_copy
        image_copy = TF.adjust_brightness(image_copy, brightness)
        image_copy = TF.adjust_contrast(image_copy, contrast)
        image_copy = TF.adjust_saturation(image_copy, saturation)
        image_copy = TF.adjust_hue(image_copy, hue)
        
        # convert image back to numpy array
        image_copy = np.array(image_copy)

        return {'image': image_copy, 'keypoints': key_pts_copy}



class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))
        
        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}

# new transform to rotate the images
class RandomRotate(object):

    def __init__(self, rot_angle):
        # assert isinstance(prob, (int, float))
        self.rot_angle = rot_angle
        
    def __call__(self, sample):

        # get random rotation angle each time the transform is called
        self.theta = np.random.uniform(-self.rot_angle, self.rot_angle) 

        image, key_pts = sample['image'], sample['keypoints']
            
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # get rotation matrix for key points
        theta_r = np.radians(self.theta)
        kpt_rotation = np.array([[np.cos(theta_r), np.sin(theta_r)],
                                 [-np.sin(theta_r), np.cos(theta_r)]])

        # get centroid of the key points
        centroid = key_pts.sum(axis=0) // (key_pts.shape[0])

        # get rotation matrix for image
        img_rotation = cv2.getRotationMatrix2D(tuple(centroid), 
                                               angle=self.theta, 
                                               scale=1)

        # rotate image and key points about key point centroid
        w, h = image.shape[:2]
        rotated_img = cv2.warpAffine(image, img_rotation, (w,h))

        rotated_kpt = centroid + ((key_pts - centroid) @ kpt_rotation.T)

        # get rotated image key point dict
        dict_out = {'image': rotated_img, 'keypoints': rotated_kpt}

        return dict_out
