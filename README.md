# Tool-as-Interface: Learning Robot Tool Use from Human Play through Imitation Learning

<img width="90%" src="doc/teaser.png">

[Haonan Chen](http://haonan16.github.io/)<sup>1</sup>,
[Cheng Zhu](#)<sup>1</sup>,
[Yunzhu Li](https://yunzhuli.github.io/)<sup>2</sup>,
[Katherine Driggs-Campbell](https://krdc.web.illinois.edu/)<sup>1</sup>

<sup>1</sup>University of Illinois, Urbana-Champaign,
<sup>2</sup>Columbia University,


### Hardware Requirements
  - [UR5-CB3](https://www.universal-robots.com/cb3) or [UR5e](https://www.universal-robots.com/products/ur5-robot/) (with RTDE Interface)
  - Alternative: [Kinova Gen 3](https://www.kinovarobotics.com/product/gen3-robots)
* **Sensors**:
  - 2× [Intel RealSense D415](https://www.intelrealsense.com/depth-camera-d415/)
  - USB-C cables and mounting hardware
* **Control Interfaces**:
  - [3Dconnexion SpaceMouse](https://3dconnexion.com/us/product/spacemouse-wireless/) (teleoperation)
  - [GELLO Controller](https://wuphilipp.github.io/gello_site/) (teleoperation)
* **Custom Components**:
  - 3D Printed [Hammer and Nail](https://drive.google.com/drive/folders/1HnnNGtl3wS5ApkalbBFL0AzQjoxBDtIR?usp=drive_link)
  
    To install a tool on the UR5E, we provide two types of fast tool changers:
    
      - 3D-Printed [Clipper](https://drive.google.com/drive/folders/1SKQ4NMSN-Cp9kYj_B2mragblk8PEFqc_?usp=drive_link) 
        
        - Requires a [connector](https://drive.google.com/file/d/1JBD4EhKf4gAre9dwJdqgS4vPZRU5ML8w/view?usp=drive_link) to attach tools to the Clipper.  
          - Example: The hammer linked above already includes the connector.  

        The upper Clipper is connected to the Clipper base using one M4×16 screw and one M4 nut. A [clipper gasket](https://drive.google.com/file/d/11Jw-0TL7u2tzo6TFm0f1b6uPaByDueeN/view?usp=sharing) is provided for placing between the UR5E robot and the Clipper. If the gasket is chosen to use, you should use four M6×30 screws. Without the clipper gasket, four M6×24 screws will work as well. Four M6 screw gasket will be used in both conditions.
        
      - 3D Printed [Mounter](https://drive.google.com/drive/folders/1Ex7M8BWiQKhk5oJQzOgIL4AuPSCr0W8m?usp=drive_link)

        - Suitable for both 3D-printed and standard tools.  
        - Secured using one or two 3D-printed screws.  

        To connect the Mounter with UR5E robot, you should use four M6×12 screws. Four M6 screw gasket will be used here to make it tightly connected.

### Environment Setup

We recommend using [Mambaforge](https://gyithub.com/conda-forge/miniforge#mambaforge) over the standard Anaconda distribution for a faster installation process. Create your environment using:

1. Install the necessary dependencies:
    ```console
    sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf libglm-dev
    ```

2. Clone the repository:
    ```console
    git clone --recursive https://github.com/Tool-as-Interface/Tool_as_Interface.git
    cd Tool_as_Interface/
    git clone https://github.com/xinyu1205/recognize-anything.git third_party/Grounded-Segment-Anything/recognize-anything
    ```

3. Update `mamba`, and create and activate the environment:

    To update `mamba` and create the environment, use the following commands:

    ```console
    mamba install mamba=1.5.1 -n base -c conda-forge
    mamba env create -f conda_environment_real.yml
    mamba activate ti
    ```


4. Install packages:
    ```console
    # grounded sam
    export AM_I_DOCKER=False
    export BUILD_WITH_CUDA=True
    export CUDA_HOME=/usr/local/cuda-11.8

    pip install --no-build-isolation -e third_party/Grounded-Segment-Anything/GroundingDINO
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
    pip install third_party/Grounded-Segment-Anything/grounded-sam-osx/transformer_utils

    # FoundationPose
    CONDA_ENV_PATH=$(conda info --base)/envs/$(basename "$CONDA_PREFIX")
    EIGEN_PATH="$CONDA_ENV_PATH/include/eigen3"
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$EIGEN_PATH"
    cd third_party/FoundationPose
    CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 console build_all_conda.sh
    cd ../..
    ```

5. Download the checkpoints
    ```console
    bash setup_downloads.sh
    ```

### Verify the Installation
To ensure everything is installed correctly, run the following commands:

```console
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available())'
python -c 'import torchvision; print(torchvision.__version__)'
python -c "from groundingdino.util.inference import Model; from segment_anything import sam_model_registry, SamPredictor"
```




## Data Preparation

```console
mkdir -p data
```

Place the `hammer_human` dataset in the `data` folder. The directory structure should be:


To obtain the dataset, download the corresponding zip file from the following link:  
[hammer_human dataset](https://drive.google.com/file/d/1QC_gd7dLacsePPKCJomqCKge1q_jarz0/view?usp=sharing)  

Once downloaded, extract the contents into the `data` folder.

Directory structure:

```
data/
└── hammer_human/
```





## Demo, Training and Eval on a Real Robot


Activate conda environment and login to [wandb](https://wandb.ai) (if you haven't already).
```console
conda activate ti
wandb login
```

### Calibrate the multi cameras 
Adjust the arguments in the `calibrate_extrinsics` function located in `ti/real_world/multi_realsense.py` to calibrate multiple cameras. The calibration is performed using a **400×300 mm Charuco board** with a **checker size of 40 mm** (`DICT_4X4`).

The `robot_base_in_world` parameter is manually measured and tuned in `ti/real_world/multi_realsense.py`.

<img src="doc/ChArUco_Board.jpg" alt="Calibration Board" width="300">

Run the following command to perform the calibration:

```console
python ti/real_world/multi_realsense.py
```


### Collecting Demonstration Data

Start the demonstration collection script. The following script applies to both human play collection and robot demonstration collection.
- Press `C` to start recording.
- Press `S` to stop recording.
- Press `Backspace` to delete the most recent recording (confirm with `y/n`).  

#### Collect Human Play Data
Run the following script to collect human video demonstrations:

```console
python scripts/collect_human_video.py
```

Once the human play data has been collected, process the raw data using:


```console
python scripts/preprocess_human_play.py
```


#### Collect Robot Demonstration Data (Baseline)
If you want to use **GELLO**, please calibrate the **GELLO offset** using the following script:

```console
python ti/devices/gello_software/scripts/gello_get_offset.py
```
After calibration, update the YAML configuration file:

```console
ti/devices/gello_software/gello.yaml
```

For details on performing the calibration, refer to:

```console
ti/devices/gello_software/README.md
```

Once the calibration is complete, update the argument in `scripts/demo_real_ur5e.py` to select either **Spacemouse** or **GELLO**, then run the following command to collect robot demonstration data:


```console
python scripts/demo_real_ur5e.py
```


### Training
To launch training, run:


```console
python scripts/train_diffusion_policy.py \
  --config-name=train_diffusion_policy.yaml 
```

### Evaluation on a Real Robot

Modify `eval_real_ur5e_human.py` or `eval_real_ur5e.py`, then launch the evaluation script:

- Press `C` to start evaluation (handing control over to the policy).
- Press `S` to stop the current episode.

#### Evaluate Human Play-Trained Policy
For `eval_real_ur5e_human.py`, we assume that the tool and the end-effector (EEF) are rigidly attached. Therefore, the tool pose estimation only needs to be performed once.
 - Press `T` once to estimate the tool's pose in the EEF frame.  

```console
python eval_real_ur5e_human.py 
```

#### Evaluate Teleoperation-Trained Policy

```console
python eval_real_ur5e.py 
```

### Troubleshooting

If you encounter the following error:

```console
AttributeError: module 'collections' has no attribute 'MutableMapping'
```
Resolve it by installing the correct version of protobuf:


```console
pip install protobuf==3.20.1
```

## Acknowledgement
* Policy training implementation is adapted from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/tree/main).
