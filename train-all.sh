#!/bin/bash

# Clean data folder
./clean-data.sh

# Train Region Model
echo "TRAINING REGION MODEL"
/home/mkhoshdeli/torch/install/bin/th run.lua --dataset nc-r --datapath ../../data/Mina/ --model models/encoder.lua --labelHeight 28 --labelWidth 28
/home/mkhoshdeli/torch/install/bin/th run.lua --dataset nc-r --datapath ../../data/Mina/ --model models/decoder.lua --CNNEncoder model-best.net
mv model-best.net trained-models/model-region.net
rm *.net
mv *.log logs/region/

# Train Potential Model
echo "TRAINING POTENTIAL MODEL"
/home/mkhoshdeli/torch/install/bin/th run.lua --dataset nc-p --datapath ../../data/Mina/ --model models/encoder-pot.lua --labelHeight 28 --labelWidth 28
/home/mkhoshdeli/torch/install/bin/th run.lua --dataset nc-p --datapath ../../data/Mina/ --model models/decoder-pot.lua --CNNEncoder model-best.net
mv model-best.net trained-models/model-potential.net
rm *.net
mv *.log logs/potential/

# Create Predictions from Region and Potential
echo "EVALUATING REGION MODEL"
/home/mkhoshdeli/torch/install/bin/th evaluate-region.lua
echo "EVALUATING POTENTIAL MODEL"
/home/mkhoshdeli/torch/install/bin/th evaluate-potential.lua

# Create Fusion input
echo "CREATING FUSION INPUT"
python create-fusion-input.py

# Train Fusion Model
echo "TRAINING FUSION MODEL"
/home/mkhoshdeli/torch/install/bin/th run.lua --dataset nc-f --datapath ../../data/Mina/ --model models/fusion.lua 
mv model-best.net trained-models/model-fusion.net
rm *.net
mv *.log logs/fusion/

# Evaluate Fusion Model
echo "Evaluating Fusion Model"
/home/mkhoshdeli/torch/install/bin/th evaluate-fusion.lua
