<p align="center">
  <picture>
    <img alt="K-Scale Open Source Robotics" src="https://media.kscale.dev/kscale-open-source-header.png" style="max-width: 100%;">
  </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kscalelabs/gym-ksuite/main/LICENSE)
[![Discord](https://img.shields.io/discord/1224056091017478166)](https://discord.gg/k5mSvCkYQh)
[![Wiki](https://img.shields.io/badge/wiki-humanoids-black)](https://humanoids.wiki)
<br />
[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Update Stompy S3 Model](https://github.com/kscalelabs/sim/actions/workflows/update_stompy_s3.yml/badge.svg)](https://github.com/kscalelabs/sim/actions/workflows/update_stompy_s3.yml)

</div>
<h1 align="center">
    <p>K-Scale Manipulation Suite</p>
</h1>

## Gymnasium+MuJoCo Environments

<table>
  <tr>
    <td><img src="assets/solo_arm.png" width="100%" alt="KManipSoloArm Env"/></td>
    <td><img src="assets/dual_arm.png" width="100%" alt="KManipDualArm Env"/></td>
    <td><img src="assets/full_body.png" width="100%" alt="KManipTorso Env"/></td>
  </tr>
  <tr>
    <td align="center"><b>KManipSoloArm</b> environment has one 7dof arm with a 1dof gripper. <b>KManipSoloArmVision</b> has a gripper cam, a head cam, and an overhead cam.</td>
    <td align="center"><b>KManipDualArm</b> environment has two 7dof arms with 1dof grippers. <b>KManipDualArmVision</b> has 2 gripper cams, a head cam, and an overhead cam.</td>
    <td align="center"><b>KManipTorso</b> environment has a 2dof head, two 6dof arms with 1dof grippers. <b>KManipTorsoVision</b> has 2 gripper cams, a head cam, and an overhead cam.</td>
  </tr>
</table>


## Setup - Linux

clone and install dependencies

```bash
git clone https://github.com/kscalelabs/gym-kmanip.git && cd gym-kmanip
conda create -y -n gym-kmanip python=3.10 && conda activate gym-kmanip
pip install -e .
```

run tests

```bash
pip install pytest
pytest tests/test_env.py
```

## Setup - Jetson Orin AGX

no conda on arm64, just install on bare metal

```bash
sudo apt-get install libhdf5-dev
git clone https://github.com/kscalelabs/gym-kmanip.git && cd gym-kmanip
pip install -e .
```

## Usage - Basic

visualize the mujoco scene

```bash
python gym_kmanip/examples/1_view_env.py
```

record a video of the mujoco scene

```bash
python gym_kmanip/examples/2_record_video.py
```

## Usage - Recording Data

🤗 [K-Scale HuggingFace Datasets](https://huggingface.co/kscalelabs)

data is recorded via teleop, this requires additional dependencies

```bash
pip install opencv-python==4.9.0.80
pip install vuer==0.0.30
pip install rerun-sdk==0.16.0
```

start the server on the robot computer

```bash
python gym_kmanip/examples/4_record_data_teleop.py
```

start ngrok on the robot computer.

```bash
ngrok http 8012
```

open the browser app on the vr headset and go to the ngrok url

## Usage - Visualizing Data

data is visualized using rerun

```bash
rerun gym_kmanip/data/test.rrd
```

## Usage - MuJoCo Sim Visualizer

mujoco provides a nice visualizer where you can directly control the robot

[download standalone mujoco](https://github.com/google-deepmind/mujoco/releases)

```
tar -xzf ~/Downloads/mujoco-3.1.5-linux-x86_64.tar.gz -C /path/to/mujoco-3.1.5
/path/to/mujoco-3.1.5/bin/simulate gym_kmanip/assets/_env_solo_arm.xml
```

## Help Wanted

✅ solo arm w/ vision

✅ dual arm w/ vision

✅ torso w/ vision

✅ inverse kinematics using mujoco

⬜️ tune and improve ik

⬜️ recording dataset via teleop

⬜️ training policy from dataset

⬜️ evaluating policy on robot

## Dependencies

- [Gymnasium](https://gymnasium.farama.org/) is used for environment
- [MuJoCo](http://www.mujoco.org/) is used for physics simulation
- [PyTorch](https://pytorch.org/) is used for model training
- [Rerun](https://github.com/rerun-io/rerun/) is used for visualization
- [H5Py](https://docs.h5py.org/en/stable/) is used for logging datasets
- [HuggingFace](https://huggingface.co/) is used for dataset & model storage 
- [Vuer](https://github.com/vuer-ai/vuer) *teleop only* is used for visualization
- [ngrok](https://ngrok.com/download) *teleop only* is used for networking

helpful links and repos

- [dm_control](https://github.com/google-deepmind/dm_control)
- [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones)
- [gym-aloha](https://github.com/huggingface/gym-aloha)
- [lerobot](https://github.com/huggingface/lerobot)
- [universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface)
- [loco-mujoco](https://github.com/robfiras/loco-mujoco)

### Citation

```
@misc{teleop-2024,
  title={gym-kmanip},
  author={Hugo Ponte},
  year={2024},
  url={https://github.com/kscalelabs/gym-kmanip}
}
```