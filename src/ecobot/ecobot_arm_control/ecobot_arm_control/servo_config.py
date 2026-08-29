# 'servo_range' is the servo's full mechanical travel and is what maps an angle
# onto the pulse_min..pulse_max span. 'min_angle'/'max_angle' are the *joint*
# limits the arm is allowed to command within that travel (they mirror the
# limits in urdf/ecobot_arm.urdf). The two are only equal when the joint can use
# the servo's whole sweep — the shoulder cannot, so they must stay separate.
JOINTS = [
    {
        'name': 'arm_base_joint',
        'label': 'Base',
        'channel': 3,
        'home_angle': 95,
        'min_angle': 0,
        'max_angle': 220,
        'servo_range': 270,   # DS 3218
        'angle_offset': 95,   # servo angle that points the arm straight ahead
        'pulse_min': 150,
        'pulse_max': 600,
        'speed': 1,
        'move_interval_ms': 15,
    },
    {
        'name': 'arm_shoulder_joint',
        'label': 'Shoulder',
        'channel': 0,
        'home_angle': 30,
        'min_angle': 0,
        'max_angle': 125,
        'servo_range': 180,   # TD 8130MG
        'angle_offset': 0,    # UNVERIFIED — measure against the physical arm
        'pulse_min': 150,
        'pulse_max': 600,
        'speed': 1,
        'move_interval_ms': 15,
    },
    {
        'name': 'arm_elbow_joint',
        'label': 'Elbow',
        'channel': 1,
        'home_angle': 180,
        'min_angle': 0,
        'max_angle': 180,
        'servo_range': 180,   # TD 8130MG
        'angle_offset': 0,    # UNVERIFIED — measure against the physical arm
        'pulse_min': 150,
        'pulse_max': 600,
        'speed': 1,
        'move_interval_ms': 15,
    },
    {
        'name': 'arm_wrist_joint',
        'label': 'Wrist',
        'channel': 2,
        'home_angle': 25,
        'min_angle': 0,
        'max_angle': 180,     # Wrist max angle set to 180 deg
        'servo_range': 180,   # MG 996R
        'angle_offset': 0,    # Zero offset for 0..180 deg wrist travel
        'pulse_min': 150,
        'pulse_max': 600,
        'speed': 1,
        'move_interval_ms': 15,
    },
]

NUM_JOINTS = len(JOINTS)


def apply_overrides(overrides):
    """Apply per-joint runtime overrides (used by arm_manual_node /
    arm_scanner_node / arm_target_tracker at startup, loaded from a YAML
    params file) so kinematic calibration is a config-file change instead
    of a code edit.

    `overrides` maps joint name -> {field: value}. e.g.
        arm_shoulder_joint: { servo_range: 180, angle_offset: 12,
                               min_angle: 5, max_angle: 125 }
    """
    for i, j in enumerate(JOINTS):
        o = overrides.get(j['name'])
        if not o:
            continue
        for field in ('min_angle', 'max_angle', 'angle_offset',
                      'servo_range', 'pulse_min', 'pulse_max',
                      'home_angle', 'speed'):
            if field in o:
                j[field] = float(o[field])


# ArmKinematics works in a frame where base angle 0 points straight ahead
# (+x) and angles grow counter-clockwise. The servos have their own zero, so
# every IK result must be shifted into servo space before it is commanded, and
# every servo reading shifted back before it is fed to forward().

def to_servo(ik_angles):
    """IK-frame angles -> servo angles."""
    return [float(a) + JOINTS[i]['angle_offset']
            for i, a in enumerate(ik_angles)]


def to_ik(servo_angles):
    """Servo angles -> IK-frame angles."""
    return [float(a) - JOINTS[i]['angle_offset']
            for i, a in enumerate(servo_angles)]


def within_limits(servo_angles):
    """True when every servo angle sits inside its joint limits."""
    return all(
        JOINTS[i]['min_angle'] <= a <= JOINTS[i]['max_angle']
        for i, a in enumerate(servo_angles)
    )
