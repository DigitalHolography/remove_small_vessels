# Large Vessel Mask Generation Script

This script generates large vessel masks for eyeflow measures in a dataset or a single measure folder. It also provides an option to revert previously generated masks.

## Requirements

Python 3.12

CUDA 12.x (for GPU execution)

cuDNN compatible with CUDA and ONNX Runtime GPU

Python packages (see requirements.txt)

```pip install -r requirements.txt```

## Dataset structure

Excpected dataset structure:

```
dataset/
├─ YYYY-MM-DD/
│  ├─ measure1/
│  ├─ optional_subfolder/
│  │  ├─ measure2/
│  └─ ...
├─ YYYY-MM-DD/
│  ├─ measure1/
│  └─ ...
└─ ...
```

## Script Usage

1. Single Folder Mode (default)

Run the script on a single measure folder:

```python script.py path/to/measure_folder```

Revert masks:

```python script.py path/to/measure_folder --revert```

2. Dataset Mode

Run the script on all measure folders in a dataset:

```python script.py --dataset path/to/dataset```

Revert masks for the entire dataset:

```python script.py --dataset path/to/dataset --revert```

## Result

For each folder, new masks are created as forced Masks : ```eyeflow/forceMaskArtery.png``` and ```eyeflow/forceMaskVein.png```.
During the next eyeflow processing, if the paramater ```ForcedMasks``` in the *input_EF_params.json* equals 1 or -1, they will be chosen as masks.
If forced Masks already exist, they will be renamed ```eyeflow/forceMaskArtery_full.png``` and ```eyeflow/forceMaskVein_full.png```

When reverting, original forced masks will retrieve their names, and large masks will be renamed ```eyeflow/forceMaskArtery_large.png``` and ```eyeflow/forceMaskVein_large.png```

## Notes

The script automatically downloads the ONNX model for optic disc detection if it is not present.

By default, the script runs in single-folder mode if a positional folder argument is given.

Do not specify both --folder and --dataset at the same time; the script will throw an error.

The --revert option removes previously generated masks instead of generating new ones.