import math


class RobotParams:
    def __init__(self, tread: float, wheel_radius: float,
                 reduction_ratio: float, encoder_resolution: int):
        self.tread = tread
        self.wheel_radius = wheel_radius
        self.reduction_ratio = reduction_ratio
        self.encoder_resolution = encoder_resolution

    def rpm_to_rads(self, rpm: float) -> float:
        return rpm * 2.0 * math.pi / 60.0

    def rads_to_rpm(self, rads: float) -> float:
        return rads * 60.0 / (2.0 * math.pi)


CUGOV4_PARAMS = RobotParams(
    tread=0.376,
    wheel_radius=0.03858,
    reduction_ratio=20.0,
    encoder_resolution=30,
)

CUGOV3I_PARAMS = RobotParams(
    tread=0.380,
    wheel_radius=0.03858,
    reduction_ratio=15.0,
    encoder_resolution=24,
)

UNAJU_PARAMS = RobotParams(
    tread=0.145,
    wheel_radius=0.0284,
    reduction_ratio=74.83,
    encoder_resolution=48,
)


def twist_to_rpm(linear_x: float, angular_z: float, params: RobotParams) -> tuple:
    r = params.wheel_radius
    tread = params.tread
    l_rad = linear_x / r - tread * angular_z / (2.0 * r)
    r_rad = linear_x / r + tread * angular_z / (2.0 * r)
    l_rpm = params.rads_to_rpm(l_rad)
    r_rpm = params.rads_to_rpm(r_rad)
    return l_rpm, r_rpm


def encoder_delta_to_twist(diff_l: int, diff_r: int, dt: float, params: RobotParams) -> tuple:
    if dt <= 0:
        return 0.0, 0.0
    r = params.wheel_radius
    rad_per_count = 2.0 * math.pi / (params.encoder_resolution * params.reduction_ratio)
    l_vel = diff_l * rad_per_count * r / dt
    r_vel = diff_r * rad_per_count * r / dt
    linear_x = (l_vel + r_vel) / 2.0
    angular_z = (r_vel - l_vel) / params.tread
    return linear_x, angular_z
