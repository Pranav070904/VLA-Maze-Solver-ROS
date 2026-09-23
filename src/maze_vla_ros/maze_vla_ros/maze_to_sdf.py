import numpy as np

WALL_SDF = """<sdf version="1.8">
<model name="{name}"><static>true</static><pose>{x} {y} 0.25 0 0 0</pose>
  <link name="link">
    <collision name="col"><geometry><box><size>1 1 0.5</size></box></geometry></collision>
    <visual name="vis"><geometry><box><size>1 1 0.5</size></box></geometry>
      <material><ambient>0 0 0 1</ambient><diffuse>0 0 0 1</diffuse><specular>0 0 0 1</specular></material></visual>
  </link>
</model></sdf>"""

GOAL_SDF = """<sdf version="1.8">
<model name="{name}"><static>true</static><pose>{x} {y} 0.15 0 0 0</pose>
  <link name="link">
    <visual name="vis"><geometry><cylinder><radius>0.35</radius><length>0.3</length></cylinder></geometry>
      <material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>{emissive}</material></visual>
  </link>
</model></sdf>"""

FLOOR_SDF = """<sdf version="1.8">
<model name="maze_floor"><static>true</static><pose>{cx} {cy} -0.01 0 0 0</pose>
  <link name="link">
    <collision name="col"><geometry><box><size>{w} {h} 0.02</size></box></geometry></collision>
    <visual name="vis"><geometry><box><size>{w} {h} 0.02</size></box></geometry>
      <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse>{emissive}</material></visual>
  </link>
</model></sdf>"""


def cell_to_world(row, col, cell_size=1.0):
    """Grid (row, col) -> Gazebo (x, y). Flip row so the maze renders right-side-up."""
    return col * cell_size, -row * cell_size


def wall_models(grid, cell_size=1.0):
    models = []
    for r, c in zip(*np.where(grid == 1)):
        x, y = cell_to_world(r, c, cell_size)
        name = f"wall_{r}_{c}"
        models.append((name, WALL_SDF.format(name=name, x=x, y=y), x, y, 0.25))
    return models


def _emissive_tag(enabled, r=1, g=1, b=1):
    # emissive makes a material glow that color regardless of incoming light,
    # which is exactly what keeps it flat/training-like under normal lighting
    # but also what makes it immune to shadows/lighting perturbation sweeps --
    # disable it (flat_materials=False) to let those sweeps actually register.
    return f"<emissive>{r} {g} {b} 1</emissive>" if enabled else ""


def goal_model(name, row, col, color, cell_size=1.0, flat_materials=True):
    x, y = cell_to_world(row, col, cell_size)
    r, g, b = {"red": (1, 0, 0), "green": (0, 1, 0)}[color]
    emissive = _emissive_tag(flat_materials, r, g, b)
    return name, GOAL_SDF.format(name=name, x=x, y=y, r=r, g=g, b=b, emissive=emissive), x, y, 0.15


def floor_model(grid_shape, cell_size=1.0, flat_materials=True):
    h, w = grid_shape
    cx, cy = (w - 1) * cell_size / 2, -(h - 1) * cell_size / 2
    emissive = _emissive_tag(flat_materials)
    return ("maze_floor", FLOOR_SDF.format(cx=cx, cy=cy, w=w * cell_size, h=h * cell_size, emissive=emissive),
            cx, cy, -0.01)