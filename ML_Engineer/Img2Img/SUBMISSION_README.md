## Poly Interview
## Keanu Spies (keanuspies@gmail.com)

Thank you so much for the opportunity to work on this problem and present it to you. 

Please find my response to the interview package in this directory. Namely:
- `model.py` - contains the PyTorch models for the task at hand
- `data.py` - contains a custom Dataset class along with some visualization helpers
- `training.ipynb` - contains a detailed, step-by-step training flow of the models along with some visualized results. 
- `results` - directory containing some train logs along with test results. Sub-directories are labeled with type of filter applied/trained. See training.ipynb for more details. 

Note: I hope I haven't over-simplified the problem at hand, but I did want to solve precisely what was being asked. Given a less clear prompt and a less simple filter I would have opted to train something more generative in a pix2pix etc manner (since we have paired image2image data), however I felt this circumstance definitely did not call for such. 

My model was trained on the [tiny-imagenet-200](https://huggingface.co/datasets/Maysee/tiny-imagenet) dataset

## The Sobel filter

I did some digging into the filter itself [Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator) and found that most often what is called the "Sobel Operation" is two Sobel filters Gx and Gy applied to an image and their combined magnitude as the output. That being said I wanted to allow for this in my model training as well hence, in `model.py` you will see a `DoubleFilter` class along with the `SingleFilter` class. This `DoubleFilter` returns the magnitude or both filters applied to the image only. 

## The random filter

I also allowed for a random filter to be applied in both the single and double formats seen in the Sobel filter. I also allowed for the control of normalization of the output. 


## Question answers

1. What if the image is really large or not of a standard size?
    <b> Since my model is a single conv layer that outputs the same size as the input this should not be a problem for training. The sobel filter generally stays the same size and has the same values for images of all shapes and sizes so the learned filter should still work. <b>


2. What should occur at the edges of the image?
    <b> Images should be padded to conserve the image shape we desire </b>

3. Are you using a fully convolutional architecture?
    <b> Yes (with an activation/ non-linearity), since we are learning single kernels this can be done with a single conv layer. </b>

4. Are there optimizations built into your framework of choice (e.g. Pytorch) that can make this fast?
    <b> - This can be trained on a GPU/with cuda, I trained this on my M1 chip. 
        - (Resonably) Larger batch sizes could also be used to speed up training/inference. 
        - Torch has some pruning methods (see below) to decrease size and still work performatively well. 
        - Instead of the dataset structure I created where the filter is applied at time of retrieval I can save each filter to file and read instead. This would improve training speed some as well. 
    </b>

5. What if you wanted to optimize specifically for model size?
    <b>
        While this model is already effectively as small as possible, in general I would prune my model (to have sparse connections) using something like the torch.nn.utils.prune library. 
    </b>

6. How do you know when training is complete?
    <b> a. When train and val loss meet/are both minimized we know that out model is well trained. This could allow for early stopping etc.
        b. When valid loss is minimized this is generally a good stop, since we are assuming the validation set is a good sample of unseen data. 
     </b>

7. What is the benefit of a deeper model in this case? When might there be a benefit for a deeper model (not with a sobel kernel but generally when thinking about image to image transformations)?
    <b>
        A deeper model could learn more complex filters/more than 1 filter other than my single/double filter method. It could also learn transformations that are not only filters, although this might require a different architechure as well (similar to a pix2pix or GAN method). 
    </b>


(extra) What are the limitations of (arbitrary kernel approach)?
<b>
My approach can only learn a single or double kernel filter or 3x3 size. This excludes a wide variety of filters that one could apply to the image. This is not to mention that filters might undergo a variety of non-linearities/activations after they are applied which I have not coded for. If I wanted to train something entirely generalizable I could train a true neural image to image model that would learn any applied filter such as [pix2pix](https://phillipi.github.io/pix2pix/)

Some extra limitations: 
- This method requires paired images, if I wanted to learn unpaired transforms I would have to use something like a [CycleGAN](https://junyanz.github.io/CycleGAN/) to acchieve the transforms. 
- My method has to be trained each time to learn the transfer. There are methods that are able to apply an arbitrary style (or set of designated styles) to an arbitrary image (a la style transfer) eg. [Huang and Belongie, 2017](https://arxiv.org/pdf/1703.06868.pdf). 
</b> 