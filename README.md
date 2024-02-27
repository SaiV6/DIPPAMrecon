# DIPPAMrecon

# Deep Image Prior Reconstruction

This project implements a Deep Image Prior (DIP) based approach for image reconstruction. It utilizes a convolutional network to reconstruct or denoise images without requiring clean training data.

## Requirements

- Python 3.x
- PyTorch
- PIL (Python Imaging Library)
- Matplotlib
- CUDA-enabled GPU (optional but recommended for faster processing)

## Installation

Clone this repository and navigate into the project directory. Install the dependencies using pip:

```bash
pip install torch torchvision pillow matplotlib
```

There is a requirements file you can review that I have used in virtual environments. 
Use the following line to download in the requirements:

```bash
pip install -r requirements.txt
```

Then you can run the following line in your terminal to start reconstruction:

```bash
python main.py --image "/path/to/your/image.png" --iterations 500 --alpha 0.8 --num_refinement_passes 3 --accumulation_steps 4 --save_path "output/path.png"
```

You can change the parameters for what you woudl like but the parameters above should give you the optimal results.
If more reconstruction is needed, add more iterations and num_refinement_passes.
If still not working, message me at sv258@duke.edu and I will attempt to fix it.

## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
