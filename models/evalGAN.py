import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm

'''
Code sourced from:
https://www.kaggle.com/code/nicholaslincoln/comparing-pix2pix-cyclegan-with-inception-score
'''

class EvalGAN:
    def __init__(self, model, real_images, fake_images):
        self.real_images = real_images
        self.fake_images = fake_images
        self.model = model
        self.model.to('cuda')
        self.model.eval()
        
    def preprocess_input(self, img):
        input_image = Image.open(img)
        processed_img = input_image.resize(299)
        return processed_img
    
    def calculate_fid(self):
        """
        Calculates the Frechet Inception Distance (FID) score.
        Implemented from: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
        """
        # calculate activations
        with torch.no_grad():
            act1 = self.model(self.real_images).cpu().detach()
            act2 = self.model(self.fake_images).cpu().detach()
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = torch.sum(((mu1 - mu2)**2.0))
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid