# Transformation Matrices - Robot_v2

Homogeneous transformation matrices between consecutive frames.
Convention: URDF RPY (XYZ extrinsic / ZYX intrinsic).

## Notation

### Frames

| Index | Link |
|-------|------|
| $L_{0}$ | base_link |
| $L_{1}$ | Hip_pitch_right |
| $L_{2}$ | Hip_pitch_left_2 |
| $L_{3}$ | Hip_roll_right |
| $L_{4}$ | Hip_roll_left_3 |
| $L_{5}$ | Hip_yaw_right |
| $L_{6}$ | Hip_yaw_left_3 |
| $L_{7}$ | Knee_right |
| $L_{8}$ | Knee_left_3 |
| $L_{9}$ | Ankle_right |
| $L_{10}$ | Ankle_left_2 |

### Joint Variables

| Variable | Joint | Type | From | To |
|----------|-------|------|------|----|
| $q_{1}$ | base_link_hip_pitch_right_joint | revolute (rad) | $L_{0}$ | $L_{1}$ |
| $q_{2}$ | base_link_hip_pitch_left_joint | revolute (rad) | $L_{0}$ | $L_{2}$ |
| $q_{3}$ | hip_pitch_hip_roll_right_joint | revolute (rad) | $L_{1}$ | $L_{3}$ |
| $q_{4}$ | hip_pitch_hip_roll_left_joint | revolute (rad) | $L_{2}$ | $L_{4}$ |
| $q_{5}$ | hip_roll_hip_yaw_right_joint | revolute (rad) | $L_{3}$ | $L_{5}$ |
| $q_{6}$ | hip_roll_hip_yaw_left_joint | revolute (rad) | $L_{4}$ | $L_{6}$ |
| $q_{7}$ | hip_yaw_knee_right_joint | revolute (rad) | $L_{5}$ | $L_{7}$ |
| $q_{8}$ | hip_yaw_knee_left_joint | revolute (rad) | $L_{6}$ | $L_{8}$ |
| $q_{9}$ | knee_ankle_right_joint | revolute (rad) | $L_{7}$ | $L_{9}$ |
| $q_{10}$ | knee_ankle_left_joint | revolute (rad) | $L_{8}$ | $L_{10}$ |

Shorthand: $c_i = \cos(q_i)$, $s_i = \sin(q_i)$

### Kinematic Tree

```
L0: base_link
  |-- [revolute] base_link_hip_pitch_right_joint (q1)
  |   L1: Hip_pitch_right
  |     +-- [revolute] hip_pitch_hip_roll_right_joint (q3)
  |         L3: Hip_roll_right
  |           +-- [revolute] hip_roll_hip_yaw_right_joint (q5)
  |               L5: Hip_yaw_right
  |                 +-- [revolute] hip_yaw_knee_right_joint (q7)
  |                     L7: Knee_right
  |                       +-- [revolute] knee_ankle_right_joint (q9)
  |                           L9: Ankle_right
  +-- [revolute] base_link_hip_pitch_left_joint (q2)
      L2: Hip_pitch_left_2
        +-- [revolute] hip_pitch_hip_roll_left_joint (q4)
            L4: Hip_roll_left_3
              +-- [revolute] hip_roll_hip_yaw_left_joint (q6)
                  L6: Hip_yaw_left_3
                    +-- [revolute] hip_yaw_knee_left_joint (q8)
                        L8: Knee_left_3
                          +-- [revolute] knee_ankle_left_joint (q10)
                              L10: Ankle_left_2
```

## Transforms

## base_link_hip_pitch_right_joint

$L_{0}$ **base_link** -> $L_{1}$ **Hip_pitch_right** (revolute)
  Variable: $q_{1}$

- **origin xyz**: (0.05735, -0.0255, 0) m
- **origin rpy**: (0, 0, 0) rad
- **axis**: (-1, 0, 0)
- **limits**: [-0.785398, 0.785398] rad ([-45deg, 45deg])

### Local Transform

$$
T^{0}_{1}(q_{1}) = \begin{bmatrix}
1 & 0 & 0 & 0.05735 \\
0 & c_{1} & s_{1} & -0.0255 \\
0 & -s_{1} & c_{1} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## base_link_hip_pitch_left_joint

$L_{0}$ **base_link** -> $L_{2}$ **Hip_pitch_left_2** (revolute)
  Variable: $q_{2}$

- **origin xyz**: (-0.039157, -0.0255, 0) m
- **origin rpy**: (-3.141593, 0, -3.141593) rad
- **axis**: (-1, 0, 0)
- **limits**: [-0.785398, 0.785398] rad ([-45deg, 45deg])

### Local Transform

$T^{0}_{2}(q_{2}) = T_{fixed} \cdot R_{axis}(q_{2})$ where:

$$
T_{fixed} = \begin{bmatrix}
-1 & 0 & 0 & -0.039157 \\
0 & 1 & 0 & -0.0255 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{2}) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & c_{2} & s_{2} & 0 \\
0 & -s_{2} & c_{2} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## hip_pitch_hip_roll_right_joint

$L_{1}$ **Hip_pitch_right** -> $L_{3}$ **Hip_roll_right** (revolute)
  Variable: $q_{3}$

- **origin xyz**: (0.053444, -0.029434, -0.018728) m
- **origin rpy**: (3.141593, 0, -1.570796) rad
- **axis**: (0, 0, 1)
- **limits**: [-0.785398, 0.20944] rad ([-45deg, 12deg])

### Local Transform

$T^{1}_{3}(q_{3}) = T_{fixed} \cdot R_{axis}(q_{3})$ where:

$$
T_{fixed} = \begin{bmatrix}
0 & -1 & 0 & 0.053444 \\
-1 & 0 & 0 & -0.029434 \\
0 & 0 & -1 & -0.018728 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{3}) = \begin{bmatrix}
c_{3} & -s_{3} & 0 & 0 \\
s_{3} & c_{3} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## hip_pitch_hip_roll_left_joint

$L_{2}$ **Hip_pitch_left_2** -> $L_{4}$ **Hip_roll_left_3** (revolute)
  Variable: $q_{4}$

- **origin xyz**: (0.015944, -0.029434, -0.019072) m
- **origin rpy**: (-3.141593, 0, -1.570796) rad
- **axis**: (0, 0, -1)
- **limits**: [-0.20944, 0.785398] rad ([-12deg, 45deg])

### Local Transform

$T^{2}_{4}(q_{4}) = T_{fixed} \cdot R_{axis}(q_{4})$ where:

$$
T_{fixed} = \begin{bmatrix}
0 & -1 & 0 & 0.015944 \\
-1 & 0 & 0 & -0.029434 \\
0 & 0 & -1 & -0.019072 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{4}) = \begin{bmatrix}
c_{4} & s_{4} & 0 & 0 \\
-s_{4} & c_{4} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## hip_roll_hip_yaw_right_joint

$L_{3}$ **Hip_roll_right** -> $L_{5}$ **Hip_yaw_right** (revolute)
  Variable: $q_{5}$

- **origin xyz**: (0.061534, -0.004021, -0.018762) m
- **origin rpy**: (0, 0, 1.570796) rad
- **axis**: (0, -1, 0)
- **limits**: [-0.785398, 0.785398] rad ([-45deg, 45deg])

### Local Transform

$T^{3}_{5}(q_{5}) = T_{fixed} \cdot R_{axis}(q_{5})$ where:

$$
T_{fixed} = \begin{bmatrix}
0 & -1 & 0 & 0.061534 \\
1 & 0 & 0 & -0.004021 \\
0 & 0 & 1 & -0.018762 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{5}) = \begin{bmatrix}
c_{5} & 0 & -s_{5} & 0 \\
0 & 1 & 0 & 0 \\
s_{5} & 0 & c_{5} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## hip_roll_hip_yaw_left_joint

$L_{4}$ **Hip_roll_left_3** -> $L_{6}$ **Hip_yaw_left_3** (revolute)
  Variable: $q_{6}$

- **origin xyz**: (0.061534, -0.004021, -0.018738) m
- **origin rpy**: (0, 0, 1.570796) rad
- **axis**: (0, 1, 0)
- **limits**: [-0.785398, 0.785398] rad ([-45deg, 45deg])

### Local Transform

$T^{4}_{6}(q_{6}) = T_{fixed} \cdot R_{axis}(q_{6})$ where:

$$
T_{fixed} = \begin{bmatrix}
0 & -1 & 0 & 0.061534 \\
1 & 0 & 0 & -0.004021 \\
0 & 0 & 1 & -0.018738 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{6}) = \begin{bmatrix}
c_{6} & 0 & s_{6} & 0 \\
0 & 1 & 0 & 0 \\
-s_{6} & 0 & c_{6} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## hip_yaw_knee_right_joint

$L_{5}$ **Hip_yaw_right** -> $L_{7}$ **Knee_right** (revolute)
  Variable: $q_{7}$

- **origin xyz**: (0.0291, -0.068651, 0.0005) m
- **origin rpy**: (-3.141593, 0, -3.141593) rad
- **axis**: (-1, 0, 0)
- **limits**: [-0.785398, 1.570796] rad ([-45deg, 90deg])

### Local Transform

$T^{5}_{7}(q_{7}) = T_{fixed} \cdot R_{axis}(q_{7})$ where:

$$
T_{fixed} = \begin{bmatrix}
-1 & 0 & 0 & 0.0291 \\
0 & 1 & 0 & -0.068651 \\
0 & 0 & -1 & 0.0005 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{7}) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & c_{7} & s_{7} & 0 \\
0 & -s_{7} & c_{7} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## hip_yaw_knee_left_joint

$L_{6}$ **Hip_yaw_left_3** -> $L_{8}$ **Knee_left_3** (revolute)
  Variable: $q_{8}$

- **origin xyz**: (-0.0087, -0.068651, -0.0005) m
- **origin rpy**: (-3.141593, 0, -3.141593) rad
- **axis**: (-1, 0, 0)
- **limits**: [-1.570796, 0.785398] rad ([-90deg, 45deg])

### Local Transform

$T^{6}_{8}(q_{8}) = T_{fixed} \cdot R_{axis}(q_{8})$ where:

$$
T_{fixed} = \begin{bmatrix}
-1 & 0 & 0 & -0.0087 \\
0 & 1 & 0 & -0.068651 \\
0 & 0 & -1 & -0.0005 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{8}) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & c_{8} & s_{8} & 0 \\
0 & -s_{8} & c_{8} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## knee_ankle_right_joint

$L_{7}$ **Knee_right** -> $L_{9}$ **Ankle_right** (revolute)
  Variable: $q_{9}$

- **origin xyz**: (0.0375, -0.113448, 0) m
- **origin rpy**: (0, 0, -3.141593) rad
- **axis**: (-1, 0, 0)
- **limits**: [-1.047198, 1.047198] rad ([-60deg, 60deg])

### Local Transform

$T^{7}_{9}(q_{9}) = T_{fixed} \cdot R_{axis}(q_{9})$ where:

$$
T_{fixed} = \begin{bmatrix}
-1 & 0 & 0 & 0.0375 \\
0 & -1 & 0 & -0.113448 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{9}) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & c_{9} & s_{9} & 0 \\
0 & -s_{9} & c_{9} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## knee_ankle_left_joint

$L_{8}$ **Knee_left_3** -> $L_{10}$ **Ankle_left_2** (revolute)
  Variable: $q_{10}$

- **origin xyz**: (-0.0375, -0.113448, 0) m
- **origin rpy**: (0, 0, -3.141593) rad
- **axis**: (1, 0, 0)
- **limits**: [-1.047198, 1.047198] rad ([-60deg, 60deg])

### Local Transform

$T^{8}_{10}(q_{10}) = T_{fixed} \cdot R_{axis}(q_{10})$ where:

$$
T_{fixed} = \begin{bmatrix}
-1 & 0 & 0 & -0.0375 \\
0 & -1 & 0 & -0.113448 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

$$
R_{axis}(q_{10}) = \begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & c_{10} & -s_{10} & 0 \\
0 & s_{10} & c_{10} & 0 \\
0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

---

## Global Transform Chains

Transform from root $L_0$ to any link, as product of local transforms along the kinematic chain.

$$T^{0}_{3} = T^{0}_{1}(q_{1}) \cdot T^{1}_{3}(q_{3})\quad (L_0 \to L_{3}: \text{Hip_roll_right})$$

$$T^{0}_{4} = T^{0}_{2}(q_{2}) \cdot T^{2}_{4}(q_{4})\quad (L_0 \to L_{4}: \text{Hip_roll_left_3})$$

$$T^{0}_{5} = T^{0}_{1}(q_{1}) \cdot T^{1}_{3}(q_{3}) \cdot T^{3}_{5}(q_{5})\quad (L_0 \to L_{5}: \text{Hip_yaw_right})$$

$$T^{0}_{6} = T^{0}_{2}(q_{2}) \cdot T^{2}_{4}(q_{4}) \cdot T^{4}_{6}(q_{6})\quad (L_0 \to L_{6}: \text{Hip_yaw_left_3})$$

$$T^{0}_{7} = T^{0}_{1}(q_{1}) \cdot T^{1}_{3}(q_{3}) \cdot T^{3}_{5}(q_{5}) \cdot T^{5}_{7}(q_{7})\quad (L_0 \to L_{7}: \text{Knee_right})$$

$$T^{0}_{8} = T^{0}_{2}(q_{2}) \cdot T^{2}_{4}(q_{4}) \cdot T^{4}_{6}(q_{6}) \cdot T^{6}_{8}(q_{8})\quad (L_0 \to L_{8}: \text{Knee_left_3})$$

$$T^{0}_{9} = T^{0}_{1}(q_{1}) \cdot T^{1}_{3}(q_{3}) \cdot T^{3}_{5}(q_{5}) \cdot T^{5}_{7}(q_{7}) \cdot T^{7}_{9}(q_{9})\quad (L_0 \to L_{9}: \text{Ankle_right})$$

$$T^{0}_{10} = T^{0}_{2}(q_{2}) \cdot T^{2}_{4}(q_{4}) \cdot T^{4}_{6}(q_{6}) \cdot T^{6}_{8}(q_{8}) \cdot T^{8}_{10}(q_{10})\quad (L_0 \to L_{10}: \text{Ankle_left_2})$$

