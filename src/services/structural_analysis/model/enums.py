from enum import Enum


class ElementType(str, Enum):
    FRAME_2D = "frame_2d"
    FRAME_2D_HINGED = "frame_2d_hinged"
    FRAME_3D = "frame_3d"
    TRUSS_2D = "truss_2d"
    TRUSS_3D = "truss_3d"
    PLANE_STRESS_Q4 = "plane_stress_q4"
    PLANE_STRESS_Q4_GAUSS = "plane_stress_q4_gauss"
    PLANE_STRESS_Q9 = "plane_stress_q9"
    SHELL_DKQ = "shell_dkq"
    SHELL_MITC4 = "shell_mitc4"
    BRICK_H8 = "brick_h8"
    TETRA_T4 = "tetra_t4"
    SPRING = "spring"


class LoadType(str, Enum):
    DEAD = "dead"
    LIVE = "live"
    WIND = "wind"
    SNOW = "snow"
    EARTHQUAKE_X = "earthquake_x"
    EARTHQUAKE_Y = "earthquake_y"
    TEMPERATURE = "temperature"
    OTHER = "other"


class RestraintAxis(str, Enum):
    UX = "ux"
    UY = "uy"
    UZ = "uz"
    RX = "rx"
    RY = "ry"
    RZ = "rz"
