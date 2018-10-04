#!/usr/local/bin/python
# coding: utf-8
import logging
import math
from typing import Optional, Tuple

import bpy
import numpy as np

from mdsea import loghandler
from mdsea.analytics import Vis
from mdsea.core import SysManager
from mdsea.vis import blender_materials as b_materials, \
    blender_objetcs as b_objects

log = logging.getLogger(__name__)
log.addHandler(loghandler)

Material = bpy.types.Material
Tuple3 = Tuple[float, float, float]
Tuple4 = Tuple[float, float, float, float]


class BlenderAnimation(Vis):
    def __init__(self, sm: SysManager,
                 frame_step: int = 1) -> None:
        super(BlenderAnimation, self).__init__(sm, frame_step)
        
        # ==============================================================
        # ---  Parce user arguments/settings
        # ==============================================================
        
        # Blender particle limit per emiter: 10,000,000
        max_particles = 10000000
        if self.sm.NUM_PARTICLES > max_particles:
            raise ValueError(
                f"Blender only suports {max_particles} particles per emiter. "
                f"You are requesting {self.sm.NUM_PARTICLES} particles.")
        
        # ==============================================================
        # ---  Other variables
        # ==============================================================
        
        # Save scene in class scope
        self.scene = bpy.data.scenes["Scene"]
        
        # Objects
        self.particle_sys = None
        self.ref_particle = None
        
        # Start at frame 0
        self.frame_num = 0
        
        # ==============================================================
        # ---  Method calls
        # ==============================================================
        
        # Start by saving the .blend file
        self.save()
        
        # Frames 'n stuff
        self.set_framelim()
        self._set_sceneframe(self.frame_num)
        
        # Remove default objects (Cube and Light)
        self._rm_defaultobjects()
    
    # ==================================================================
    # ---  Private Methods
    # ==================================================================
    
    @staticmethod
    def _rm_defaultobjects() -> None:
        for obj_name in ("Cube", "Lamp"):
            bpy.data.objects[obj_name].select = True
            bpy.ops.object.delete()
    
    def _set_sceneframe(self, frame: int, advance: bool = True) -> None:
        self.scene.frame_set(frame=frame)
        if advance:
            self.frame_num += 1
    
    # ==================================================================
    # ---  User/Public Methods
    # ==================================================================
    
    def quick_setup(self, engine: Optional[str] = None) -> None:
        """ Miscellaneous setups and preferences """
        # Set render engine before dealing with materials!
        if engine is not None:
            self.engine = engine
        # Others:
        self.set_view()
        self.set_world()
        self.setup_camera()
        self.set_render_preferences()
    
    def set_render_preferences(self,
                               resolution_x: float = 1080,
                               resolution_y: float = 1080,
                               resolution_percentage: float = 100,
                               cycles_samples: int = 256,
                               render_raytrace: bool = False
                               ) -> None:
        # Dimensions
        self.scene.render.resolution_x = resolution_x
        self.scene.render.resolution_y = resolution_y
        self.scene.render.resolution_percentage = resolution_percentage
        self.scene.render.use_border = True
        self.scene.render.use_crop_to_border = True
        # Output
        self.scene.render.filepath = f"{self.sm.png_path}/img00"
        self.scene.render.image_settings.compression = 100
        self.scene.render.use_placeholder = True
        if self.engine == 'CYCLES':
            # Sampling
            self.scene.cycles.samples = cycles_samples
            self.scene.cycles.preview_samples = 8
            self.scene.cycles.sample_clamp_direct = 0
            self.scene.cycles.sample_clamp_indirect = 1
            # Light paths
            self.scene.cycles.max_bounces = 10
            self.scene.cycles.min_bounces = 4
            self.scene.cycles.transparent_max_bounces = 1
            self.scene.cycles.transparent_min_bounces = 0
            self.scene.cycles.transmission_bounces = 10
            self.scene.cycles.diffuse_bounces = 4
            self.scene.cycles.glossy_bounces = 6
            self.scene.cycles.volume_bounces = 0
            self.scene.cycles.use_transparent_shadows = True
            self.scene.cycles.caustics_reflective = True
            self.scene.cycles.caustics_refractive = True
            self.scene.cycles.blur_glossy = 1
            # Film
            self.scene.cycles.film_transparent = True
            self.scene.cycles.pixel_filter_type = 'BOX'
            # Performance
            self.scene.render.use_persistent_data = False
            self.scene.cycles.debug_use_spatial_splits = True
            self.scene.cycles.debug_bvh_type = "STATIC_BVH"
        elif self.engine == 'BLENDER_RENDER':
            self.scene.render.use_raytrace = render_raytrace
    
    # TODO: not used
    @staticmethod
    def update_particle_color(pid, frame, vx, vy, vz,
                              speed_limit) -> None:
        factor = _get_speed_factor(vx, vy, vz, speed_limit)
        mat = bpy.data.materials[f"Particle{pid}"]
        mat.node_tree.nodes["Mix Shader"].inputs['Fac'].default_value = factor
        mat.node_tree.nodes["Mix Shader"].inputs['Fac'].keyframe_insert(
            "default_value", frame=frame)
    
    def create_ref_particle(self,
                            mat: Optional[Material] = None) -> None:
        """ Create a reference particle for the particle system. """
        # Create a new "UV Sphere" object
        loc = -1000 * self.sm.LEN_BOX  # in a galaxy far far away...
        bpy.ops.mesh.primitive_uv_sphere_add(location=(loc, loc, loc))
        # Save particle object in class scope
        self.ref_particle = bpy.context.object
        self.ref_particle.name = "Reference Particle"
        bpy.ops.object.shade_smooth()
        # Set material
        if mat is None:
            mat = b_materials.particle(self.engine)
        b_materials.set_material(self.ref_particle, mat)
        # Set sphere object dimentions
        diameter = 2 * self.sm.RADIUS_PARTICLE
        self.ref_particle.dimensions = (diameter, diameter, diameter)
    
    def create_particle_system(self,
                               mat: Optional[Material] = None
                               ) -> None:
        
        # Create particles (you need to create a reference
        # particle before creating the particle system!)
        if self.ref_particle is None:
            self.create_ref_particle(mat)
        
        # Create a plane
        bpy.ops.mesh.primitive_plane_add(location=(0, 0, 0))
        self.particle_sys = bpy.context.object
        self.particle_sys.name = "Particle System"
        self.particle_sys.dimensions = (0, 0, 0)
        
        # Add a particle system to the plane
        bpy.ops.object.particle_system_add()
        
        ############
        # SETTINGS #
        ############
        
        psys_settings = bpy.data.particles[0]
        
        # Emission
        psys_settings.count = self.sm.NUM_PARTICLES
        psys_settings.frame_start = 0
        psys_settings.frame_end = 0
        psys_settings.lifetime = self.num_frames + 1
        
        # Velocity
        psys_settings.normal_factor = 0
        
        # Physics
        psys_settings.timestep = 0
        
        # Render
        psys_settings.render_type = "OBJECT"
        psys_settings.particle_size = 1
        psys_settings.use_scale_dupli = True
        psys_settings.use_render_emitter = False
        psys_settings.dupli_object = self.ref_particle
        
        # Field Weights (set all to zero)
        psys_settings.effector_weights.gravity = 0
        psys_settings.effector_weights.all = 0
        psys_settings.effector_weights.force = 0
        psys_settings.effector_weights.vortex = 0
        psys_settings.effector_weights.magnetic = 0
        psys_settings.effector_weights.wind = 0
        psys_settings.effector_weights.curve_guide = 0
        psys_settings.effector_weights.texture = 0
        psys_settings.effector_weights.smokeflow = 0
        psys_settings.effector_weights.harmonic = 0
        psys_settings.effector_weights.charge = 0
        psys_settings.effector_weights.lennardjones = 0
        psys_settings.effector_weights.turbulence = 0
        psys_settings.effector_weights.drag = 0
        psys_settings.effector_weights.boid = 0
        
        # We need to update the scene to make
        # Blender aware of these changes.
        self.update_scene()
    
    def save(self, path: Optional[str] = None) -> None:
        """ Save .blend file. """
        if path is None:
            path = f"{self.sm.blender_path}/{self.sm.SIM_ID}.blend"
        # noinspection PyCallByClass,PyTypeChecker
        bpy.ops.wm.save_as_mainfile(filepath=path)
    
    def bake(self) -> None:
        # Select and activate particle system
        self.particle_sys.select = True
        self.scene.objects.active = self.particle_sys
        # Bake from cache
        cc = bpy.context.copy()
        cc["point_cache"] = self.particle_sys.particle_systems[0].point_cache
        bpy.ops.ptcache.bake_from_cache(cc)
    
    def run(self) -> None:
        
        # Animate particles step-by-step
        for self.step in np.arange(0, self.sm.STEPS, self.frame_step):
            
            if self.step == 0:
                pass
            elif self.step == self.sm.STEPS:
                break
            
            # Start by setting the scene frame
            self._set_sceneframe(self.frame_num)
            
            # The location coordinates have to be a flat array
            coords = self.r_vecs[self.step].flatten()
            
            self.particle_sys.particle_systems[0].particles.foreach_set(
                "location", coords)
            
            self.update_scene()
        
        # We're setting the frame one last time
        # so that we can see the changes
        self._set_sceneframe(self.frame_num, advance=False)
        self._set_sceneframe(self.num_frames, advance=False)
        
        # Bake simulation
        self.bake()
        
        # Save .blend file again
        self.save()
        
        # TODO: Use disk cache (not working...?!)
        # self.particle_sys.particle_systems[0] \
        #     .point_cache.use_disk_cache = True
        
        self.update_scene()
    
    @staticmethod
    def set_world(horizon_color: Tuple3 = (0.5, 0.7, 0.8)) -> None:
        # if horizon_color is None:
        #     horizon_color = (0.5, 0.5, 0.55)
        bpy.data.worlds["World"].horizon_color = horizon_color
        bpy.data.worlds["World"].cycles_visibility.camera = False
    
    @staticmethod
    def set_view(shade: str = 'RENDERED',
                 perspective: str = 'CAMERA'
                 ) -> None:
        # Set view perspective
        area = next(a for a in bpy.context.screen.areas if a.type == 'VIEW_3D')
        area.spaces[0].region_3d.view_perspective = perspective
        # Set viewport shading
        space = next(s for s in area.spaces if s.type == 'VIEW_3D')
        space.viewport_shade = shade
    
    def update_scene(self) -> None:
        """ Update scene. """
        self.scene.update()
    
    @property
    def engine(self):
        """ Blender rendering engine. """
        return self.scene.render.engine
    
    @engine.setter
    def engine(self, engine: str):
        """ Set the rendering engine. """
        self.scene.render.engine = engine
    
    def set_framelim(self, start: int = 1,
                     end: Optional[int] = None) -> None:
        """ Set the frame limits ('start frame' and 'end frame'). """
        if end is None:
            end = self.num_frames
        self.scene.frame_start = start
        self.scene.frame_end = end
        # Lock frame selection to range:
        self.scene.lock_frame_selection_to_range = True
    
    def setup_camera(self,
                     loc: Optional[Tuple3] = None,
                     rot: Optional[Tuple3] = None,
                     clip_end: Optional[float] = None,
                     focal_lenght: float = 45,
                     sensor_width: float = 32) -> None:
        if loc is None:
            loc = (2.5 * self.sm.LEN_BOX,
                   -0.75 * self.sm.LEN_BOX,
                   1.25 * self.sm.LEN_BOX)
        if rot is None:
            rot = (math.radians(69.5), 0, math.radians(57))
        if clip_end is None:
            clip_end = 1000 * self.sm.LEN_BOX
        
        camera = bpy.data.objects["Camera"]
        camera.location = loc
        camera.rotation_euler = rot
        bpy.data.cameras["Camera"].lens = focal_lenght
        bpy.data.cameras["Camera"].clip_end = clip_end
        bpy.data.cameras["Camera"].sensor_width = sensor_width
    
    def add_light(self, where=None, loc: Optional[Tuple3] = None,
                  rot: Optional[Tuple3] = None) -> None:
        """ Add the default light object. """
        assert where in (None, 'front', 'left', 'above', 'diagonal')
        
        # shorthand
        len_ = self.sm.LEN_BOX
        
        if where is None or where == 'front':
            where = 'front'
        elif where == 'left':
            loc = (0.5 * len_, -4 * len_, 3 * len_)
            rot = (math.radians(65), 0, 0)
        elif where == 'above':
            loc = (0.5 * len_, 0.5 * len_, 7 * len_)
            rot = (0, 0, math.radians(90))
        elif where == 'diagonal':
            loc = (5 * len_, -4 * len_, 3 * len_)
            rot = (math.radians(65), math.radians(60), 0)
        
        # Create light object
        b_objects.light(self.engine, id_=where, len_box=len_, loc=loc, rot=rot)
    
    def add_floor(self):
        """ Add the default floor object. """
        b_objects.floor(self.engine, len_box=self.sm.LEN_BOX)
    
    def add_glasswalls(self):
        """ Add the default glass walls (glass cube) objects. """
        b_objects.glasswalls(self.engine, len_box=self.sm.LEN_BOX,
                             except_=['bottom'])
    
    # noinspection PyCallByClass,PyTypeChecker
    @staticmethod
    def render(opengl: bool = False, ) -> None:
        if opengl:
            bpy.ops.render.opengl(animation=True)
            return
        bpy.ops.render.render(animation=True)
