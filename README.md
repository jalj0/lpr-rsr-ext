# Number Plate Recognition Through Improved Super Resolution and OCR Model
Original Paper: [Super-Resolution of License Plate Images Using
 Attention Modules and Sub-Pixel Convolution Layers](https://doi.org/10.1016/j.cag.2023.05.005).
## Updates
1. Leveraged [Deformable Convolution](https://arxiv.org/pdf/1703.06211) layers to enhance the model’s performance.
2. Formulated a new loss function based on the SSIM metric, termed as `Dissimilarity Loss`.
## Custom Dataset Fine-tune
### 1. Dataset Structure
1. Make a text file 'split.txt' which will contain the path for HR and LR with type, all seperated by ';'.
```python
dataset/HR/img891.jpg;dataset/LR/img891.jpg;training
dataset/HR/img185.jpg;dataset/LR/img185.jpg;validation
dataset/HR/img089.jpg;dataset/LR/img089.jpg;testing
```
2. Run the following code to fine-tune
```python
# Set mode to '1' to resume training & select your best model
python training.py -t ./dataset/split.txt -s ./save -b 2 -m 1 --model /home1/jalaj_l/Proposed/save/bestpretrainedmodel.pt
```
## Training from Start
1. Run the following code to start training from scratch
```python
# Set mode to '1' to resume training & select your best model
python training.py -t ./dataset/split.txt -s ./save -b 2 -m 0
```
## Testing the Model
1. Run the following code for testing the performance of the model
```python
# Remove mode argument for testing
python testing.py -t ./dataset/split.txt -s ./save -b 2 --model /home1/jalaj_l/Proposed/save/bestpretrainedmodel.pt
```

## Known Issues and Solutions

### 1. \_\_dataset\_\_.py
* DataFrame to CSV Export Issue

Issue: You may encounter a error in `load_dataset()`.
```python
Error in load_dataset: list index out of range
```

Solution: Diagnose the code line by line using some `print()` operation.

Example:
```python
# Since in the 'label.txt' no type is given therefore bypass 'type'. And now the first line is 'plate' & the 2nd line is 'layout'.

# tp = fp.readlines()[0].split(':')[1].replace('\n', '').replace(' ', '')
tp = line[2]	#bypass type     
plate = fp.readlines()[0].split(':')[0].replace('\n', '').replace(' ', '')
layout = fp.readlines()[0].split(':')[1].replace('\n', '').replace(' ', '')
```

### 2. testing.py
* Structural Similarity Index (SSIM) Error

Issue: You may encounter a `ValueError` when using the `structural_similarity` function from `skimage.metrics` if the `win_size` exceeds the image dimensions.

Solution:

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
* DataFrame to CSV Export Issue

Issue: When exporting a DataFrame to CSV using pandas, you might encounter a TypeError if using the argument `line_terminator` instead of `lineterminator`.

Solution: Replace line_terminator with lineterminator to specify the line terminator for the CSV output.

Example:
```python
# Use lineterminator instead of line_terminator
df.to_csv('output.csv', lineterminator='\n')
```
* DataFrame average operation on all column

Issue:
```python
TypeError: Could not convert [...] to numeric:
df.loc['Average'] = df.mean(axis=0)
```

Solution:
In your DataFrame, columns like 'Type', 'Layout', 'GT Plate', 'file', and predictions contain string values, which cannot be averaged. Use below code instead.
```python
df.loc['Average'] = df.select_dtypes(include=[np.number]).mean(axis=0)
```

### 3. training.py
* Import Issue

Issue:
If you want to use MobileNetV2, please upgrade your tensorflow version and you can use as it mentioned in the documentation as `from NetSr_v1 import MobileNetV2` might not work.

Solution:

1. For Google Colab and latest version of tensorflow, Use: `!pip install keras_applications` will install keras-applications >= 1.0.8 For tensorflow version >= 2.5.0
2. use from `keras.applications.mobilenet_v2 import MobileNetV2`.

### 4. eval_csv.py
* Levenshtein distance problem

Issue:
```python
In eval_csv.py line no 20, errors = Levenshtein.distance(a, b)
```

Solution: This worked for me:
```python
from training import SSIMLoss
	...
	def eval_char(gt,sr):
		for a, b in zip(gt, sr):
        	errors = criterion.levenshtein(str(a), str(b))
		...
```
* levenshtein() problem

Issue:
```python
TypeError: 'float' object is not subscriptable
```

Solution: A float value is being passed to the levenshtein() function, which expects strings to compare character by character. This likely means that some values in gt (ground truth) or sr (super-resolved predictions) are not strings but floats, and the function tries to perform character-by-character comparisons on a float, which is not possible. After here and there you can convert a&b to str(a) & str(b). But you will find NaN which is one of possible prediction.
```python
# Replace NaN values with empty strings
gt = gt.fillna('')
sr = sr.fillna('')
```

## Citation
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

