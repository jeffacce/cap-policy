import time
from collections import deque
import cv2
import numpy as np
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
import hydra
from scipy.spatial.transform import Rotation as R
import os
import xml.etree.ElementTree as ET
import gdown
import zipfile
import random

def init_model_loss_fn(cfg):
    model = hydra.utils.instantiate(cfg.model).to(cfg.device)
    model_weight_pth = cfg.get("model_weight_pth")

    if model_weight_pth is None:
        raise ValueError("Model weight path is not specified in the config.")

    checkpoint = torch.load(
        model_weight_pth, map_location=cfg.device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    loss_fn = hydra.utils.instantiate(cfg.loss_fn, model=model)
    loss_fn.load_state_dict(checkpoint["loss_fn"])
    loss_fn = loss_fn.to(cfg.device)

    return model, loss_fn


def action_tensor_to_matrix(action_tensor, rot_unit):
    affine = np.eye(4)
    if rot_unit == "euler":
        r = R.from_euler("xyz", action_tensor[3:6], degrees=False)
    elif rot_unit == "axis":
        r = R.from_rotvec(action_tensor[3:6])
    else:
        raise NotImplementedError
    affine[:3, :3] = r.as_matrix()
    affine[:3, -1] = action_tensor[:3]
    return affine


def get_objects(root_folder):
    results = []

    for dirpath, _, filenames in os.walk(root_folder):
        if "model.xml" in filenames:
            full_path = os.path.join(dirpath, "model.xml")
            parent_folder = os.path.basename(dirpath)
            results.append((parent_folder, full_path))

    return results

def add_object_to_scene(scene_xml_content, include_file_path):
    root = ET.fromstring(scene_xml_content)
    new_include = ET.Element("include", file=include_file_path)
    includes = root.findall("include")
    if includes:
        index = list(root).index(includes[-1]) + 1
        root.insert(index, new_include)
    else:
        root.insert(0, new_include)
    return ET.tostring(root, encoding="unicode")


def download_and_extract_zip(gdrive_link, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_id = gdrive_link.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    zip_path = os.path.join(output_dir, "temp_download.zip")

    gdown.download(download_url, zip_path, quiet=False)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)

def world_to_pixel(p_world, model, data, cam_name, img_width, img_height):
    xmat = data.camera(cam_name).xmat
    R_wc = xmat.reshape(3, 3)
    pos = data.camera(cam_name).xpos
    R_cw = R_wc.T
    t_cw = -R_cw @ pos

    p_cam = R_cw @ p_world + t_cw
    x, y, z = p_cam
    if z >= 0:
        return [0.5, 0.5]

    fovy_rad = model.camera(cam_name).fovy[0] * np.pi / 180.0
    aspect = img_width / img_height
    fy = 0.5 * img_height / np.tan(fovy_rad / 2)
    fx = fy * aspect
    cx, cy = img_width / 2, img_height / 2

    u = fx * (x / -z) + cx
    v = cy - fy * (y / -z)

    return [u / img_width, v / img_height]

def world_to_camera(p_world, model, data, cam_name):

    # mujoco camera axis
    # x-axis to right
    # y-axis to up
    # z-axis to backwards

    # desired axis
    # x-axis to left
    # y-axis to forward
    # z-axis to down

    T_world_camera = np.eye(4)
    T_world_camera[:3, :3] = data.camera(cam_name).xmat.reshape(3, 3).copy()
    T_world_camera[:3, -1] = data.camera(cam_name).xpos.copy()

    R_world_camera = T_world_camera[:3, :3]
    R_camera_world = R_world_camera.T

    p_camera = R_camera_world @ (p_world - T_world_camera[:3, -1])

    desired_axis = np.array([[-1, 0,  0],
                  [ 0, 0, -1],
                  [ 0,-1,  0]])

    return desired_axis @ p_camera

def set_global_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)

def euler_to_mat_xyz(rx, ry, rz):
    cx = torch.cos(rx)
    sx = torch.sin(rx)
    cy = torch.cos(ry)
    sy = torch.sin(ry)
    cz = torch.cos(rz)
    sz = torch.sin(rz)
    B = rx.shape[0]
    Rmats = torch.empty(B, 3, 3, device=rx.device)
    Rmats[:, 0, 0] = cy * cz
    Rmats[:, 0, 1] = -cy * sz
    Rmats[:, 0, 2] = sy
    Rmats[:, 1, 0] = cx * sz + cz * sx * sy
    Rmats[:, 1, 1] = cx * cz - sx * sy * sz
    Rmats[:, 1, 2] = -cy * sx
    Rmats[:, 2, 0] = sx * sz - cx * cz * sy
    Rmats[:, 2, 1] = cz * sx + cx * sy * sz
    Rmats[:, 2, 2] = cx * cy
    return Rmats


def batch_build_affines(actions, rot_unit="euler"):
    B = actions.shape[0]
    affines = torch.eye(4, device=actions.device).unsqueeze(0).repeat(B, 1, 1)
    affines[:, :3, 3] = actions[:, :3]
    if rot_unit == "euler":
        rx = actions[:, 3]
        ry = actions[:, 4]
        rz = actions[:, 5]
        Rmats = euler_to_mat_xyz(rx, ry, rz)
    elif rot_unit == "axis":
        rvec = actions[:, 3:6]
        theta = torch.norm(rvec, dim=1, keepdim=True)
        eps = 1e-8
        theta = torch.where(theta < eps, torch.ones_like(theta) * eps, theta)
        k = rvec / theta
        B_val = B
        K = torch.zeros(B_val, 3, 3, device=actions.device)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] = k[:, 0]
        I = torch.eye(3, device=actions.device).unsqueeze(0).expand(B, -1, -1)
        sin_theta = torch.sin(theta).view(B, 1, 1)
        cos_theta = torch.cos(theta).view(B, 1, 1)
        Rmats = I + sin_theta * K + (1 - cos_theta) * torch.bmm(K, K)
    else:
        raise NotImplementedError("Unsupported rotation unit.")
    affines[:, :3, :3] = Rmats
    return affines

def batch_apply_transform(affines, M, M_inv):
    B = affines.shape[0]
    M_batch = M.unsqueeze(0).expand(B, -1, -1)
    M_inv_batch = M_inv.unsqueeze(0).expand(B, -1, -1)
    out = torch.bmm(torch.bmm(M_batch, affines), M_inv_batch)
    return out

def batch_extract_euler_xyz(Rmats):
    r00 = Rmats[:, 0, 0]
    r10 = Rmats[:, 1, 0]
    r20 = Rmats[:, 2, 0]
    r21 = Rmats[:, 2, 1]
    r22 = Rmats[:, 2, 2]
    sy = torch.sqrt(r00**2 + r10**2)
    singular = sy < 1e-6
    x = torch.atan2(r21, r22)
    y = torch.atan2(-r20, sy)
    z = torch.atan2(r10, r00)
    x_sing = torch.atan2(-Rmats[:, 1, 2], Rmats[:, 1, 1])
    z_sing = torch.zeros_like(z)
    x = torch.where(singular, x_sing, x)
    z = torch.where(singular, z_sing, z)
    return torch.stack([x, y, z], dim=1)