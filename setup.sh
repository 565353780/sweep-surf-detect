cd ..
git clone https://github.com/565353780/base-trainer.git
git clone https://github.com/565353780/point-cept.git

cd base-trainer
./setup.sh

cd ../point-cept
./setup.sh

pip install flash-attn --no-build-isolation

pip install numpy matplotlib
