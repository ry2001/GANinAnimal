import random
import torch

class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []
            self.fake_labels = []

    def query_cond(self, images, fake_labels, real_labels):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        
        the first 50 images will be randomly paired, the others will pair with the real labels if have in the pool
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images, fake_labels
        return_images = []
        return_labels = []
        for idx, image in enumerate(images):
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
                
                self.fake_labels.append(fake_labels[idx])
                return_labels.append(fake_labels[idx])
            else:
                selection = self.pool_filter(real_labels)[idx]
                if len(selection) != 0:
                    random_id = random.choice(self.pool_filter(real_labels)[idx])
                else:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                tmp = self.images[random_id].clone()
                self.images[random_id] = image
                return_images.append(tmp)
                return_labels.append(self.fake_labels[random_id])
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        assert len(return_images) == len(return_labels), 'Images and labels size are different'
        return (return_images, return_labels)
    
    def pool_filter(self, labels):
        idx_choice = []
        for rlabel in labels:
            label_idx = []
            for idx, flabel in enumerate(self.fake_labels):
                if rlabel == flabel:
                    label_idx.append(idx)
            idx_choice.append(label_idx)
        return idx_choice
       
    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
    
