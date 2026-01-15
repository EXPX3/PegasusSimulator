"""
| File: lidar.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| Modified for Isaac Sim 5.1.0 - Auto-Config Discovery
"""
__all__ = ["Lidar"]

import os
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
import omni.kit.commands
from pxr import UsdGeom
import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.graphical_sensors import GraphicalSensor
from isaacsim.sensors.rtx import LidarRtx

class Lidar(GraphicalSensor):

    def __init__(self, lidar_name, config={}):
        super().__init__(sensor_type="Lidar", update_rate=config.get("frequency", 60.0)) 

        self._lidar_name = lidar_name
        self._stage_prim_path = ""

        # Position and Orientation
        self._position = config.get("position", np.array([0.075, 0.24, 0.075]))
        euler_orient = config.get("orientation", np.array([0, 340, 180]))
        quat_xyzw = Rotation.from_euler("ZYX", euler_orient, degrees=True).as_quat()
        self._orientation = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # 1. RESOLVE CONFIG PATH AUTOMATICALLY
        # The user provides "Velodyne_VLS128", but the file is at ".../Velodyne/Velodyne_VLS128.json"
        raw_config_name = config.get("sensor_configuration", "Velodyne_VLS128")
        self._config_file = self._resolve_lidar_config_path(raw_config_name)
        
        self._variant = config.get("variant", "Example_Rotary")
        self._show_render = config.get("show_render", True)
        self._sensor_attributes = config.get("sensor_attributes", {"omni:sensor:Core:scanRateBaseHz": self._update_rate})
        self._sensor = None

    def _resolve_lidar_config_path(self, config_name):
        """
        Searches for the JSON config file inside the isaacsim.sensors.rtx extension.
        Returns the absolute path if found, or the original name if not.
        """
        # If the user already provided a full path or a vendor path (Velodyne/Velodyne_VLS128), use it.
        if "/" in config_name or "\\" in config_name or config_name.endswith(".json"):
            return config_name

        # Get the path to the RTX sensors extension
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = ext_manager.get_enabled_extension_id("isaacsim.sensors.rtx")
        if not ext_id:
            print(f"[Lidar] Warning: isaacsim.sensors.rtx extension not found!")
            return config_name

        ext_path = ext_manager.get_extension_path(ext_id)
        config_root = os.path.join(ext_path, "data", "lidar_configs")

        # Recursively search for the file (e.g. Velodyne_VLS128.json)
        target_file = f"{config_name}.json"
        for root, dirs, files in os.walk(config_root):
            if target_file in files:
                full_path = os.path.join(root, target_file)
                print(f"[Lidar] Resolved config '{config_name}' to: {full_path}")
                return full_path

        print(f"[Lidar] Warning: Could not find config '{config_name}' in {config_root}. Using default.")
        return config_name

    def initialize(self, vehicle):
        super().initialize(vehicle)
        
        # 2. Define the Intended Path
        target_path = f"{self._vehicle.prim_path}/body/{self._lidar_name}"
        
        # 3. Ensure Parent Exists
        stage = omni.usd.get_context().get_stage()
        parent_path = f"{self._vehicle.prim_path}/body"
        if not stage.GetPrimAtPath(parent_path):
            UsdGeom.Xform.Define(stage, parent_path)

        # 4. Create Sensor with ABSOLUTE Config Path
        self._sensor = LidarRtx(
            prim_path=target_path,
            translation=self._position,
            orientation=self._orientation,
            config_file_name=self._config_file, # Now holds the full path
            variant=self._variant,
            **self._sensor_attributes
        )
        
        # 5. Path Correction (Just in case)
        current_path = self._sensor.prim_path
        if current_path != target_path:
            omni.kit.commands.execute("MovePrim", path_from=current_path, path_to=target_path)
            self._stage_prim_path = target_path
        else:
            self._stage_prim_path = current_path

    def start(self):
        if self._show_render:
            try:
                hydra_texture = rep.create.render_product(self._stage_prim_path, [1, 1], name="IsaacLidar")
                writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
                v_id = getattr(self._vehicle, '_vehicle_id', 0)
                topic_name = f"/drone{v_id}/{self._lidar_name}/pointcloud"
                writer.initialize(topicName=topic_name, frameId=f"{self._lidar_name}")
                writer.attach([hydra_texture])
            except Exception as e:
                print(f"[Lidar] Error initializing writer: {e}")

    @property
    def state(self):
        return self._state
    
    @GraphicalSensor.update_at_rate
    def update(self, state: State, dt: float):
        self._state = {
            "lidar_name": self._lidar_name, 
            "stage_prim_path": str(self._stage_prim_path)
        }
        return self._state