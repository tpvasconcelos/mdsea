#!/usr/local/bin/python
# coding: utf-8

"""
Blender objetcs.

"""

import logging
import math
from typing import Iterable, List, Optional, Tuple, Union

import bpy

from mdsea import loghandler
from mdsea.vis import blender_materials as materials

log = logging.getLogger(__name__)
log.addHandler(loghandler)

Object = bpy.types.Object
Material = bpy.types.Material
Tuple3 = Tuple[float, float, float]


def light(engine: str,
          id_: str = "",
          len_box: Optional[float] = None,
          mat: Optional[Material] = None,
          loc: Optional[Tuple3] = None,
          rot: Optional[Tuple3] = None,
          dim: Optional[Tuple3] = None
          ) -> Object:
    """ Light object. """
    
    if loc is None:
        loc = (5 * len_box, 0.5 * len_box, 3 * len_box)
    if rot is None:
        rot = (0, math.radians(65), 0)
    
    if engine == 'BLENDER_RENDER':
        return _render_light(id_, loc, rot)
    
    # ---  CYCLES  ---
    
    if dim is None:
        dim = (3 * len_box, 3 * len_box, 0)
    if mat is None:
        mat = materials.light()
    
    bpy.ops.mesh.primitive_plane_add(location=loc, rotation=rot)
    obj = bpy.context.object
    obj.name = f"Light-{id_}"
    obj.dimensions = dim
    # if engine == 'CYCLES':
    obj.cycles_visibility.camera = False
    
    materials.set_material(obj, mat)
    
    return obj


def _render_light(id_, loc: Tuple3, rot: Tuple3) -> Object:
    """ Light object (render engine). """
    dist = math.sqrt(sum(c ** 2 for c in loc))
    if id_ == 'above':
        bpy.ops.object.lamp_add(type='SUN', location=loc, rotation=rot)
        obj = bpy.context.object
        obj.data.energy = 0.25
    elif id_ == 'diagonal':
        bpy.ops.object.lamp_add(type='POINT', location=loc, rotation=rot)
        obj = bpy.context.object
        obj.data.energy = 3 * dist / 48
        # obj.data.spot_blend = 1.
        # obj.data.spot_size = math.radians(180)
    else:
        bpy.ops.object.lamp_add(type='POINT', location=loc, rotation=rot)
        obj = bpy.context.object
        obj.data.energy = 1 * dist / 24
    
    obj.name = f"Light-{id_}"
    obj.data.distance = 6 * dist
    obj.data.shadow_method = "NOSHADOW"
    
    return obj


def glasswalls(engine: str, len_box: float,
               mat: Optional[Material] = None,
               thickness: Optional[float] = None,
               which: Union[Iterable[str], str] = 'all',
               except_: Optional[Iterable[str]] = None
               ) -> List[Object]:
    """ Glass wall object. """
    
    if mat is None:
        mat = materials.glasswall(engine)
    
    if thickness is None:
        thickness = 0.01 * len_box
    
    # noinspection PyUnusedLocal
    wall_bottom = {
        "rotation": (0, 0, 0),
        "location": (len_box / 2,
                     len_box / 2,
                     -thickness / 2),
        "dimensions": (len_box + 2 * thickness,
                       len_box + 2 * thickness,
                       thickness)
        }
    
    wall_top = {
        "rotation": (0, 0, 0),
        "location": (len_box / 2,
                     len_box / 2,
                     len_box + (thickness / 2)),
        "dimensions": (len_box + 2 * thickness,
                       len_box + 2 * thickness,
                       thickness)
        }
    
    wall_front = {
        "rotation": (math.radians(90), 0, 0),
        "location": (len_box / 2,
                     -thickness / 2,
                     len_box / 2),
        "dimensions": (len_box,
                       len_box,
                       thickness)
        }
    
    wall_back = {
        "rotation": (math.radians(90), 0, 0),
        "location": (len_box / 2,
                     len_box + thickness / 2,
                     len_box / 2),
        "dimensions": (len_box,
                       len_box,
                       thickness)
        }
    
    wall_left = {
        "rotation": (0, math.radians(90), 0),
        "location": (-thickness / 2,
                     len_box / 2,
                     len_box / 2),
        "dimensions": (len_box,
                       len_box + 2 * thickness,
                       thickness)
        }
    
    wall_right = {
        "rotation": (0, math.radians(90), 0),
        "location": (len_box + (thickness / 2),
                     len_box / 2,
                     len_box / 2),
        "dimensions": (len_box,
                       len_box + 2 * thickness,
                       thickness)
        }
    
    walls_table = {
        'bottom': wall_bottom,
        'right': wall_right,
        'front': wall_front,
        'back': wall_back,
        'left': wall_left,
        'top': wall_top
        }
    
    #  which
    if which == 'all':
        which = walls_table.keys()
    elif which is None:
        which = []
    
    #  except_
    if except_ == 'all':
        except_ = walls_table.keys()
    elif except_ is None:
        except_ = []
    
    walls = []
    for w in which:
        if w not in except_:
            walls.append(walls_table[w])
    
    objs = []
    for i, wall in enumerate(walls):
        bpy.ops.mesh.primitive_cube_add(location=wall["location"],
                                        rotation=wall["rotation"])
        obj = bpy.context.object
        obj.name = f"Wall-{i}"
        obj.dimensions = wall["dimensions"]
        materials.set_material(obj, mat)
        objs.append(obj)
    
    return objs


def floor(engine: str, len_box: float,
          mat: Optional[Material] = None,
          loc: Optional[Tuple3] = None,
          dim: Optional[Tuple3] = None
          ) -> Object:
    """ Floor object. """
    
    if mat is None:
        mat = materials.floor(engine)
    if loc is None:
        loc = (0, 0, 0)
    if dim is None:
        dim = (1000 * len_box, 1000 * len_box, 0)
    
    bpy.ops.mesh.primitive_plane_add(location=loc)
    obj = bpy.context.object
    obj.name = "Floor"
    obj.dimensions = dim
    
    materials.set_material(obj, mat)
    
    return obj
