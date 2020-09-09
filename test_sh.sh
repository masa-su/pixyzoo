env_name="pixyz_py38"
cd DMM
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$env_name --execute --inplace dmm.ipynb
cd ../DynaNet
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$env_name --execute --inplace dynanet.ipynb
cd ../FactorVAE
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$env_name --execute --inplace factorvae-baseline.ipynb
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$env_name --execute --inplace factorvae.ipynb
cd ../GQN
python train.py
cd ../TD-VAE
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$env_name --execute --inplace train.ipynb
cd ../VIB
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$env_name --execute --inplace baseline.ipynb
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$env_name --execute --inplace vib.ipynb
cd ../VRNN
jupyter nbconvert --to notebook --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=$env_name --execute --inplace vrnn.ipynb
