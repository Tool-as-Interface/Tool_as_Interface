#!/bin/bash

# Function to download a file if it doesn't already exist
download_if_not_exists() {
    local url=$1
    local output_path=$2

    if [ ! -f "$output_path" ]; then
        echo "Downloading $output_path..."
        wget -q --show-progress -O "$output_path" "$url"
        echo "$output_path downloaded successfully."
    else
        echo "$output_path already exists. Skipping download."
    fi
}

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p ti/utils/Grounded_SAM/
mkdir -p third_party/FoundationPose/weights/

# Download GroundingDINO and SAM checkpoints
echo "Starting GroundingDINO and SAM checkpoints downloads..."
download_if_not_exists "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" "ti/utils/Grounded_SAM/groundingdino_swint_ogc.pth"
download_if_not_exists "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" "ti/utils/Grounded_SAM/sam_vit_h_4b8939.pth"
download_if_not_exists "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" "ti/utils/Grounded_SAM/groundingdino_swinb_cogcoor.pth"
download_if_not_exists "https://huggingface.co/lkeab/hq-sam/resolve/67ab82412bc794d5ce2e9799b8b6a3c0a8cfe1d2/sam_hq_vit_h.pth" "ti/utils/Grounded_SAM/sam_hq_vit_h.pth"



# Download FoundationPose weights only if the directory is empty
echo "Checking for FoundationPose weights..."
if [ -z "$(ls -A third_party/FoundationPose/weights)" ]; then
    echo "Downloading FoundationPose weights..."
    gdown --folder "https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i" -O "third_party/FoundationPose/weights"
    echo "FoundationPose weights downloaded successfully."
else
    echo "FoundationPose weights already exist. Skipping download."
fi


echo "All checkpoints have been successfully downloaded."
