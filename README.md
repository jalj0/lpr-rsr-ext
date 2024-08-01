# Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers
[paper:](https://doi.org/10.1016/j.cag.2023.05.005).


## Known Issues and Solutions

- \_\_dataset\_\_.py

### 1. DataFrame to CSV Export Issue

#### Issue:
You may encounter a error in `load_dataset()`.

#### Solution:
1. Diagnose the code line by line using some `print()` operation.

Example:
```python
# Use lineterminator instead of line_terminator
df.to_csv('output.csv', lineterminator='\n')
```

__testing.py__

### 1. Structural Similarity Index (SSIM) Error

#### Issue:
You may encounter a `ValueError` when using the `structural_similarity` function from `skimage.metrics` if the `win_size` exceeds the image dimensions.

#### Solution:
1. **Ensure Image Size**: Make sure your images are at least 7x7 pixels.
2. **Specify `win_size`**: Pass an odd `win_size` value that is less than or equal to the smaller dimension of your images.
3. **Set `channel_axis`**: For multichannel images (e.g., RGB), set `channel_axis` to the axis number corresponding to the channels (typically `channel_axis=2` for RGB).

Example:
```python
from skimage.metrics import structural_similarity as ssim

# Adjust parameters as necessary
SSIM_param = {
    ...
    'multichannel':True,    #No longer needed, ignoring it solves my problem
    ...
}
# set channel_axis to the axis number corresponding to the channels:    -> SSIM(channel_axis=2)
psnr_ssim[1].append(ssim(imgHR, imgSR, **SSIM_param, channel_axis = 2))    #
```

### 2. DataFrame to CSV Export Issue

#### Issue:
When exporting a DataFrame to CSV using pandas, you might encounter a TypeError if using the argument `line_terminator` instead of `lineterminator`.

#### Solution:
1. Replace line_terminator with lineterminator to specify the line terminator for the CSV output.

Example:
```python
# Use lineterminator instead of line_terminator
df.to_csv('output.csv', lineterminator='\n')
```

- __training.py__

### 1. Import Issue

#### Issue:
If you want to use MobileNetV2, please upgrade your tensorflow version and you can use as it mentioned in the documentation as `from NetSr_v1 import MobileNetV2` might not work.

#### Solution:
1. For Google Colab and latest version of tensorflow, Use: `!pip install keras_applications` will install keras-applications >= 1.0.8 For tensorflow version >= 2.5.0
2. use from `keras.applications.mobilenet_v2 import MobileNetV2`.
    
# Citation

* V. Nascimento, R. Laroca, J. A. Lambert, W. R. Schwartz, D. Menotti, “Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers,” in *Computers & Graphics*, vol. 113, pp. 69-76, 2023. [[Science Direct]](https://doi.org/10.1016/j.cag.2023.05.005) [[arXiv]](https://arxiv.org/abs/2305.17313)

```
@article{nascimento2023super,
  title = {Super-Resolution of License Plate Images Using Attention Modules and Sub-Pixel Convolution Layers},
  author = {V. {Nascimento} and R. {Laroca} and J. A. {Lambert} and W. R. {Schwartz} and D. {Menotti}},
  year = {2023},
  journal = {Computers \& Graphics},
  volume = {113},
  number = {},
  pages = {69-76},
  doi = {10.1016/j.cag.2023.05.005},
  issn = {0097-8493},
  keywords = {License plate recognition, Super-resolution, Attention modules, Sub-pixel convolution layers}
}
```

Please show support to this repository.

