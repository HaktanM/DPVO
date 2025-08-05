import yaml
import numpy as np

def load_yaml(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)

def get_T_cam0_to_cam1(calib_file):
    calib = load_yaml(calib_file)

    # Load T_imu_cam for cam0 and cam1
    T_imu_cam0 = np.array(calib['cam0']['T_imu_cam'])
    T_imu_cam1 = np.array(calib['cam1']['T_imu_cam'])

    # Compute T_cam0_to_cam1
    T_cam1_to_imu = np.linalg.inv(T_imu_cam1)  # inverse of T_imu_cam1
    T_cam0_to_cam1 = T_cam1_to_imu @ T_imu_cam0

    return T_cam0_to_cam1

if __name__ == "__main__":
    T = get_T_cam0_to_cam1("calib/euroc.yaml")
    print("T_cam0_to_cam1:\n", T)
