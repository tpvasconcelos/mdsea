"""
Blender materials.

"""

import logging
import math
import random

import bpy
from mdsea.helpers import Tuple3, Tuple4

log = logging.getLogger(__name__)

Material = bpy.types.Material


# ======================================================================
# ---  Helpers
# ======================================================================


def set_material(object_, mat) -> None:
    """ Set a material to an object. """
    object_.data.materials.clear()
    object_.data.materials.append(mat)


def _get_speed_factor(vx, vy, vz, speed_limit) -> float:
    """ TODO: not used """
    speed_ratio = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2) / speed_limit
    if speed_ratio > 1.0:
        speed_ratio = 1.0
    # speed_ratio *= 0.85
    return speed_ratio


def _get_randomcolor() -> Tuple4:
    """ Get a random color. """
    rgb_high = [0.7, 0.8, 0.9, 1.0, 1.0]
    rgb_low = [0.1, 0.2, 0.3, 0.4]
    c_main3 = [
        (random.choice(rgb_high), 0, 0, 1),
        (0, random.choice(rgb_high), 0, 1),
        (0, 0, random.choice(rgb_high), 1),
    ]
    c_hh_comb = [
        (random.choice(rgb_high), random.choice(rgb_high), 0, 1),
        (random.choice(rgb_high), 0, random.choice(rgb_high), 1),
        (0, random.choice(rgb_high), random.choice(rgb_high), 1),
    ]
    c_hl_comb = [
        (random.choice(rgb_high), random.choice(rgb_low), 0, 1),
        (random.choice(rgb_high), 0, random.choice(rgb_low), 1),
        (0, random.choice(rgb_high), random.choice(rgb_low), 1),
    ]
    c_lh_comb = [
        (random.choice(rgb_low), random.choice(rgb_high), 0, 1),
        (random.choice(rgb_low), 0, random.choice(rgb_high), 1),
        (0, random.choice(rgb_low), random.choice(rgb_high), 1),
    ]
    palette = [*c_main3, *c_hh_comb, *c_hl_comb, *c_lh_comb]
    return random.choice(palette)


# ======================================================================
# ---  Particle
# ======================================================================


def particle(
    engine,
    particle_id="",
    color: Tuple4 = (1, 0.4, 0.1, 1),
    random_color: bool = False,
    color_temp: bool = False,
    vx=None,
    vy=None,
    vz=None,
    speed_limit=None,
) -> Material:
    """ Particle material. """
    mat = bpy.data.materials.new(f"Particle{particle_id}")

    # FIXME(tpvasconcelos): Use different colors within a particle system
    # if color_temp == 'temperature':
    #     factor = _get_speed_factor(vx, vy, vz, speed_limit)

    if random_color:
        color = _get_randomcolor()

    if engine == "BLENDER_RENDER":
        return _render_particle(mat, color[:-1])
    return _cycles_particle(mat, color)


def _cycles_particle(mat: Material, color: Tuple4) -> Material:
    """ Particle material (cycles engine). """

    # Nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Create a mix shader
    mixer = nodes.new(type="ShaderNodeMixShader")
    layerweight = nodes.new(type="ShaderNodeLayerWeight")
    layerweight.inputs["Blend"].default_value = 0.3

    # create input node (DIFFUSE)
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = color
    diffuse.inputs["Color"].keyframe_insert("default_value", frame=0)
    diffuse.inputs["Roughness"].default_value = 0

    # Create input node (GLOSSY)
    glossy = nodes.new(type="ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = (1, 1, 1, 1)
    glossy.inputs["Color"].keyframe_insert("default_value", frame=0)
    glossy.inputs["Roughness"].default_value = 0.2

    # Link nodes
    links.new(layerweight.outputs["Fresnel"], mixer.inputs[0])
    links.new(diffuse.outputs["BSDF"], mixer.inputs[1])
    links.new(glossy.outputs["BSDF"], mixer.inputs[2])
    links.new(mixer.outputs["Shader"], nodes["Material Output"].inputs["Surface"])

    return mat


def _render_particle(mat: Material, color: Tuple3) -> Material:
    """ Particle material (render engine). """
    mat.diffuse_color = color
    mat.diffuse_intensity = 1
    mat.specular_intensity = 0.25
    mat.emit = 0.01
    mat.specular_hardness = 30
    return mat


# ======================================================================
# ---  Glass wall
# ======================================================================


def glasswall(engine) -> Material:
    """ Glass wall material. """
    mat = bpy.data.materials.new("GlassWall")
    color = (1, 1, 1, 1)
    ior = 1.25
    if engine == "BLENDER_RENDER":
        return _render_glasswall(mat, color[:-1], ior)
    return _cycles_glasswall(mat, color, ior)


def _cycles_glasswall(mat: Material, color: Tuple4, ior: float) -> Material:
    """ Glass wall material (cycles engine). """
    # Nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    # create input node
    nodes.new(type="ShaderNodeBsdfGlass")
    node_input = nodes["Material Output"].inputs["Surface"]
    # create output node
    node_output = nodes["Glass BSDF"].outputs["BSDF"]
    # link nodes
    mat.node_tree.links.new(node_input, node_output)
    # settings
    nodes["Glass BSDF"].inputs["Color"].default_value = color
    nodes["Glass BSDF"].inputs["IOR"].default_value = ior
    nodes["Glass BSDF"].inputs["Roughness"].default_value = 0.001
    return mat


def _render_glasswall(mat: Material, color: Tuple4, ior: float) -> Material:
    """ Glass wall material (render engine). """

    # Diffuse
    mat.diffuse_color = color
    mat.diffuse_intensity = 1.0

    # Specular
    mat.specular_intensity = 1.0
    mat.specular_hardness = 255

    # Transparency
    mat.use_transparency = True
    mat.alpha = 0.1
    mat.specular_alpha = 0.8
    mat.raytrace_transparency.fresnel = 1.37

    # Mirror
    # mat.raytrace_mirror.use = True
    # mat.raytrace_mirror.fresnel = 2.5
    # mat.raytrace_mirror.use = True
    # mat.raytrace_mirror.reflect_factor = 0.1

    mat.specular_ior = ior
    return mat


# ======================================================================
# ---  Floor
# ======================================================================


def floor(engine) -> Material:
    """ Floor material (cycles engine). """
    mat = bpy.data.materials.new("Floor")
    if engine == "BLENDER_RENDER":
        color = (0.6, 0.9, 1, 1)
        return _render_floor(mat, color[:-1])
    color = (1, 1, 1, 1)
    return _cycles_floor(mat, color)


def _cycles_floor(mat: Material, color: Tuple4) -> Material:
    """ Floor material (cycles engine). """

    # Nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Create input node (DIFFUSE)
    diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = color
    diffuse.inputs["Roughness"].default_value = 0.1

    # Create input node (GLOSSY)
    glossy = nodes.new(type="ShaderNodeBsdfGlossy")
    glossy.inputs["Color"].default_value = color
    glossy.inputs["Roughness"].default_value = 0.4

    # Create a mix shader
    mixer = nodes.new(type="ShaderNodeMixShader")
    mixer.inputs["Fac"].default_value = 0.05
    links.new(diffuse.outputs["BSDF"], mixer.inputs[1])
    links.new(glossy.outputs["BSDF"], mixer.inputs[2])
    links.new(mixer.outputs["Shader"], nodes["Material Output"].inputs["Surface"])

    return mat


def _render_floor(mat: Material, color: Tuple3) -> Material:
    """ Floor material (render engine). """
    mat.diffuse_color = color
    mat.diffuse_intensity = 1
    mat.specular_intensity = 1
    mat.translucency = 1
    mat.specular_hardness = 5
    mat.emit = 0.1
    # mat.use_shadeless = True
    return mat


# ======================================================================
# ---  Light
# ======================================================================


def light() -> Material:
    """ Light material (cycles engine only!!). """
    color = (0.9, 0.9, 0.95, 1)
    return _cycles_light(color)


def _cycles_light(color: Tuple4) -> Material:
    """ Light material (cycles engine). """
    mat = bpy.data.materials.new("BackLight")
    # NODES
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    # create input node
    nodes.new(type="ShaderNodeEmission")
    node_input = nodes["Material Output"].inputs["Surface"]
    # create output node
    node_output = nodes["Emission"].outputs["Emission"]
    # link nodes
    mat.node_tree.links.new(node_input, node_output)
    # settings
    nodes["Emission"].inputs["Color"].default_value = color
    nodes["Emission"].inputs["Strength"].default_value = 4
    return mat
