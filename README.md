# uVAE
3D-VAE: Unsupervised learning of 3D shapes from single images. 


## 3D VAE Architecture:
![_config.yml](./architecture/3dvae_architecture.png)
**Figure-1:**

## Sample shapes learned during training:

Below are the shapes of digits 2, 3, and 5 from MNIST dataset learned during training.
![Learning Digit 2](./ExampleResults/gif_files_showing_training/Digit_2_during_training.gif)
![Learning Digit 3](./ExampleResults/gif_files_showing_training/Digit_3_during_training.gif)
![Learning Digit 5](./ExampleResults/gif_files_showing_training/Digit_5_during_training.gif)
![Learning MNIST Fashion Pants](./ExampleResults/gif_files_showing_training/MNIST_Fashion_Pants_during_training.gif)
![Learning MNIST Fashion Shirt](./ExampleResults/gif_files_showing_training/MNIST_Fashion_Shirt_during_training.gif)
![Learning MNIST Fashion Shoe](./ExampleResults/gif_files_showing_training/MNIST_Fashion_Shoe_during_training.gif)

**Figure-2:**  3D shapes of digits and fashion items learned during training. From top to bottom, the shapes are the digits 2, 3, 5 and the fashion items pants, shirt and shoe.The model is trained on a combined datasets consisting of MNIST, MNIST Fashion, CelebA, and several categories of ModelNet40, and hence we see that the shapes resemble fashion items, or human faces at the initial period of training.



## Rendered images from learned 3D shapes during training:

|Input Data|Rendered images during training|
|:----------------------:|:--------------------------------:|
| ![](./ExampleResults/rendered_grid1/Corresponding_input_data.png) | ![](./ExampleResults/rendered_grid1/Rendered_Image_Grid_during_training_showing_items_from_all_datasets.gif) |
| ![](./ExampleResults/rendered_grid2/Corresponding_input_data.png) | ![](./ExampleResults/rendered_grid2/Rendered_Image_Grid_during_training_showing_only_MNIST_MNISTFashion_items.gif) |

**Figure-2:** Input data and corresponding rendered images from learned 3D shapes during training. The model is trained on a combined datasets consisting of MNIST, MNIST Fashion, CelebA, and several categories of ModelNet40.

## More:

For more examples, please see: [3D-VAE, uVAE, demo page](https://pilatracu.github.io/3dvae/ "3D-VAE demo page")   
