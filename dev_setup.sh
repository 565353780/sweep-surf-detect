cd ..
git clone git@github.com:565353780/base-trainer.git
git clone git@github.com:565353780/point-cept.git

cd base-trainer
./dev_setup.sh

cd ../point-cept
./dev_setup.sh

pip install flash-attn --no-build-isolation

pip install numpy matplotlib
