# Python Optical Flow: EPPM only

This package provides python bindings to CUDA accelerated EPPM optical flow.

### Fast Edge-Preserving PatchMatch for Large Displacement Optical Flow
	@inproceedings{bao2014cvpreppm,
	  title={Fast Edge-Preserving PatchMatch for Large Displacement Optical Flow},
	  author={Bao, Linchao and Yang, Qingxiong and Jin, Hailin},
	  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year={2014},
	  pages={3534-3541},
	  organization={IEEE}
	}

## Install
```
git clone https://github.com/linchaobao/EPPM ~/EPPM
ln -s ~/EPPM .
python setup.py build_ext -i
python demo.py
```
