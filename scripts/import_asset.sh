# #### Balancing Plate ####
# ./isaaclab.sh -p source/standalone/tools/convert_mesh.py \
#     /home/shuosha/Desktop/balance_assets/plates/plate_one_v0.obj \
#     /home/shuosha/Desktop/balance_assets/plates/plate_one_v0.usd \
#     --make-instanceable --collision-approximation convexDecomposition --mass 1.0

# NOTE: 
    # import torch
    # import numpy as np
    # from omni.isaac.lab.utils.math import quat_from_euler_xyz
    # init_quat_plate = quat_from_euler_xyz(torch.tensor(180*np.pi/180),torch.tensor(0*np.pi/180),torch.tensor(-90*np.pi/180))
    # quat = tuple(init_quat_plate.tolist())


#### Insertion ####
# ./isaaclab.sh -p source/standalone/tools/convert_mesh.py \
#     /home/shuosha/Desktop/insertion_assets/base/insertion_base.obj \
#     /home/shuosha/Desktop/insertion_assets/base/insertion_base.usd \
#     --make-instanceable --collision-approximation convexDecomposition --mass 1.0

# ./isaaclab.sh -p source/standalone/tools/convert_mesh.py \
#     /home/shuosha/Desktop/insertion_assets/base_smooth/insertion_base_smooth.obj \
#     /home/shuosha/Desktop/insertion_assets/base_smooth/insertion_base_smooth.usd \
#     --make-instanceable --collision-approximation convexDecomposition --mass 1.0

# ./isaaclab.sh -p source/standalone/tools/convert_mesh.py \
#     /home/shuosha/Desktop/insertion_assets/nut/nut.obj \
#     /home/shuosha/Desktop/insertion_assets/nut/nut.usd \
#     --make-instanceable --collision-approximation convexDecomposition --mass 1.0

# ./isaaclab.sh -p source/standalone/tools/convert_mesh.py \
#     /home/shuosha/Desktop/insertion_assets/nut_wide/nut_wide.obj \
#     /home/shuosha/Desktop/insertion_assets/nut_wide/nut_wide.usd \
#     --make-instanceable --collision-approximation convexDecomposition --mass 1.0

# ./isaaclab.sh -p source/standalone/tools/convert_mesh.py \
#     /home/shuosha/Desktop/insertion_assets/nut_poly/nut_poly.obj \
#     /home/shuosha/Desktop/insertion_assets/nut_poly/nut_poly.usd \
#     --make-instanceable --collision-approximation convexDecomposition --mass 1.0

# ./isaaclab.sh -p source/standalone/tools/convert_mesh.py \
#     /home/shuosha/Desktop/insertion_assets/nut_poly_wider/nut_poly_wider.obj \
#     /home/shuosha/Desktop/insertion_assets/nut_poly_wider/nut_poly_wider.usd \
#     --make-instanceable --collision-approximation convexDecomposition --mass 1.0

# ./isaaclab.sh -p source/standalone/tools/convert_mesh.py \
#     /home/shuosha/Desktop/insertion_assets/nut_poly_wide_smooth/nut_poly_wide_smooth.obj \
#     /home/shuosha/Desktop/insertion_assets/nut_poly_wide_smooth/nut_poly_wide_smooth.usd \
#     --make-instanceable --collision-approximation convexDecomposition --mass 1.0