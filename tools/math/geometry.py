from tools._backend import array_namespace


def euler_to_rotmat(yaw, pitch, roll):
    xp = array_namespace(yaw, pitch, roll)
    cy, sy = xp.cos(yaw), xp.sin(yaw)
    cp, sp = xp.cos(pitch), xp.sin(pitch)
    cr, sr = xp.cos(roll), xp.sin(roll)

    zero = xp.zeros_like(cy)
    one = xp.ones_like(cy)

    Rz = xp.stack(
        [
            xp.stack([cy, -sy, zero], axis=-1),
            xp.stack([sy, cy, zero], axis=-1),
            xp.stack([zero, zero, one], axis=-1),
        ],
        axis=-2,
    )

    Ry = xp.stack(
        [
            xp.stack([cp, zero, sp], axis=-1),
            xp.stack([zero, one, zero], axis=-1),
            xp.stack([-sp, zero, cp], axis=-1),
        ],
        axis=-2,
    )

    Rx = xp.stack(
        [
            xp.stack([one, zero, zero], axis=-1),
            xp.stack([zero, cr, -sr], axis=-1),
            xp.stack([zero, sr, cr], axis=-1),
        ],
        axis=-2,
    )

    return Rz @ Ry @ Rx


def rvec_to_rotmat(rvec, eps: float = 1e-8):
    xp = array_namespace(rvec)
    theta = xp.norm(rvec, axis=-1, keepdims=True).clamp_min(eps)
    k = rvec / theta

    K = xp.zeros((*rvec.shape[:-1], 3, 3), dlike=rvec)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]

    sin_t = xp.sin(theta)[..., None]
    cos_t = xp.cos(theta)[..., None]
    I = xp.broadcast_to(xp.eye(3, dlike=rvec), K.shape)
    
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def apply_transform(R, t, points):
    xp = array_namespace(R, t, points)
    rotated = points @ xp.transpose(R)
    return rotated + t[..., None, :]

def apply_transform_inverse(R, t, points):
    return (points - t[..., None, :]) @ R

def invert_transform(R, t):
    xp = array_namespace(R, t)
    R_inv = xp.transpose(R)
    t_inv = -(R_inv @ t[..., None])[..., 0]
    return R_inv, t_inv

def project_to_normalized(points_cam, eps: float = 1e-6):
    xp = array_namespace(points_cam)
    z = xp.maximum(points_cam[..., 2:3], eps)
    return xp.stack([
        points_cam[..., 0:1] / z,
        points_cam[..., 1:2] / z,
    ], axis=-1)


def compute_az_el(points_gimbal, eps=1e-8):
    xp = array_namespace(points_gimbal)
    x, y, z = (points_gimbal[..., i] for i in range(3))
    az = xp.atan2(x, y)
    h  = xp.maximum(xp.sqrt(x * x + y * y), eps)
    el = xp.atan2(z, h)
    return xp.stack([az, el], axis=-1)
