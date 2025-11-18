"""
| File: lidar.py
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
| Description: Creates a lidar sensor
"""
__all__ = ["Lidar"]

import numpy as np
from pxr import Gf
from isaacsim.sensors.rtx import LidarRtx # <-- Updated Import
from omni.isaac.core.utils.prims import get_prim_at_path # Utility for prim checks

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors import Sensor
from pegasus.simulator.logic.vehicles import Vehicle

# The following imports are no longer strictly needed for LidarRtx creation
# from omni.usd import get_context
# from pxr import Sdf 

class Lidar(Sensor):
    """The class that implements the Lidar sensor. This class inherits the base class Sensor.
    """
    def __init__(self, prim_path: str, config: dict = {}):
        """Initialize the Lidar class
        Args:
            prim_path (str): Path to the lidar prim. Global path when it starts with `/`, else local to vehicle prim path
            config (dict): A Dictionary that contains all the parameters for configuring the lidar - it can be empty or only have some of the parameters used by the lidar.
        Examples:
            The dictionary default parameters are
            >>> {"position": [0.0, 0.0, 0.0],           # Meters
            >>>  "yaw_offset": 0.0,                     # Degrees
            >>>  "rotation_rate": 20.0,                 # Hz (maps to scanRateBaseHz)
            >>>  "horizontal_fov": 360.0,               # Degrees (maps to horizontal:fov)
            >>>  "horizontal_resolution": 1.0,          # Degrees (maps to horizontal:resolution)
            >>>  "vertical_fov": 10.0,                  # Degrees (maps to vertical:fov)
            >>>  "vertical_resolution": 1.0,            # Degrees (maps to vertical:resolution)
            >>>  "min_range": 0.4,                      # Meters (maps to minRange)
            >>>  "max_range": 100.0,                    # Meters (maps to maxRange)
            >>>  "draw_points": False,                  # Draw lidar points where they hit an object (maps to drawPoints)
            >>>  "draw_lines": False,                   # Draw lidar ray lines (maps to drawLines)
            >>>  "high_lod": True,                      # Not directly used by LidarRtx constructor, but good to keep in config
            >>>  "fill_state: False}                    # Fill state with sensor data
        """

        # Initialize the Super class "object" attribute
        # update_rate set to rotation_rate
        super().__init__(sensor_type="Lidar", update_rate=config.get("rotation_rate", 20.0))

        # Save the id of the sensor
        self._prim_path = prim_path
        self._frame_id = prim_path.rpartition("/")[-1] # frame_id of the camera is the last prim path part after `/`

        # LidarRtx object placeholder
        self.lidar: LidarRtx = None # Updated: Hold the LidarRtx object

        # Get the lidar position relative to its parent prim
        self._position = np.array(config.get("position", [0.0, 0.0, 0.0]))
        # Note: LidarRtx uses (w, x, y, z) for quaternion, Gf.Quatd is (real, i, j, k) which is also (w, x, y, z)
        # Assuming the base class handles the vehicle rotation, we keep a default quaternion for local rotation.
        self._orientation = np.array([1.0, 0.0, 0.0, 0.0]) 

        # Get the lidar parameters
        # We collect the parameters needed for the LidarRtx constructor's sensor_attributes
        self._rotation_rate = config.get("rotation_rate", 20.0)
        self._horizontal_fov = config.get("horizontal_fov", 360.0)
        self._horizontal_resolution = config.get("horizontal_resolution", 1.0)
        self._vertical_fov = config.get("vertical_fov", 10.0)
        self._vertical_resolution = config.get("vertical_resolution", 1.0)
        self._min_range = config.get("min_range", 0.4)
        self._max_range = config.get("max_range", 100.0)
        self._draw_points = config.get("draw_points", False)
        self._draw_lines = config.get("draw_lines", False)
        
        # Note: 'yaw_offset' and 'high_lod' are not standard LidarRtx constructor arguments.
        # Yaw offset will be applied via the quaternion or initial position/orientation if needed.
        self._yaw_offset = config.get("yaw_offset", 0.0)
        self._high_lod = config.get("high_lod", True) 

        # Save the current state of the range sensor
        self._fill_state = config.get("fill_state", False)
        if self._fill_state:
            # LidarRtx gives point_cloud and distances. azimuth/zenith might not be directly available
            # We will use get_current_frame() which returns a dictionary.
            self._state = {
                "frame_id": self._frame_id,
                "data": None,
            }
        else:
            self._state = None
        
        # We need to map the Pegasus config to LidarRtx attributes
        self._rtx_attributes = {
            'omni:sensor:Core:scanRateBaseHz': self._rotation_rate,
            'horizontal:fov': self._horizontal_fov,
            'horizontal:resolution': self._horizontal_resolution,
            'vertical:fov': self._vertical_fov,
            'vertical:resolution': self._vertical_resolution,
            'minRange': self._min_range,
            'maxRange': self._max_range,
            'drawPoints': self._draw_points,
            'drawLines': self._draw_lines,
        }

    def initialize(self, vehicle: Vehicle, origin_lat, origin_lon, origin_alt):
        """Method that initializes the lidar sensor. It also initalizes the sensor latitude, longitude and
        altitude attributes as well as the vehicle that the sensor is attached to.
        
        Args:
            vehicle (Vehicle): The vehicle that this sensor is attached to.
            origin_lat (float): The latitude of the origin of the world in degrees (might get used by some sensors).
            origin_lon (float): The longitude of the origin of the world in degrees (might get used by some sensors).
            origin_alt (float): The altitude of the origin of the world relative to sea water level (might get used by some sensors).
        """
        super().initialize(vehicle, origin_lat, origin_lon, origin_alt)

        # Set the prim path for the camera (make it absolute)
        if self._prim_path[0] != '/':
            self._prim_path = f"{vehicle.prim_path}/{self._prim_path}"

        # --- RTX Lidar Creation ---
        # The LidarRtx constructor handles the USD prim creation and attribute setting.
        # We use a placeholder config_file_name as the attributes override the config.
        
        # Optional: Check if the prim already exists to avoid errors if this is a re-init
        if not get_prim_at_path(self._prim_path):
            self.lidar = LidarRtx(
                prim_path=self._prim_path,
                translation=self._position,
                orientation=self._orientation,
                config_file_name="Example_Rotary", # Use a default config name
                name=self._frame_id, # Optional: set a name for the core object
                **self._rtx_attributes, # Apply all custom attributes
            )
            # Attach the annotator needed to get point cloud data
            if self._fill_state:
                # Use the annotator from the provided example
                self.lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
            
            # Enable visualization if either drawLines or drawPoints is True
            if self._draw_points or self._draw_lines:
                self.lidar.enable_visualization()

        # Set the sensor's frame path
        self.frame_path = self._prim_path

    @property
    def state(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._state
    
    @Sensor.update_at_rate
    def update(self, state: State, dt: float):
        """
        Args:
            state (State): The current state of the vehicle.
            dt (float): The time elapsed between the previous and current function calls (s).
        Returns:
            (dict) A dictionary containing the current state of the sensor (the data produced by the sensor) or None
        """
        # If the lidar object hasn't been created (e.g., initialization failed), return None
        if not self.lidar:
            return None

        # Add the values to the dictionary and return it
        if self._fill_state:
            # LidarRtx provides all relevant data through get_current_frame()
            lidar_data = self.lidar.get_current_frame()
            
            # The dictionary returned by LidarRtx usually contains 'point_cloud', 'distances', etc.
            # We store the whole frame data under a single 'data' key for flexibility.
            self._state = {
                "frame_id": self._frame_id,
                "data": lidar_data,
                # Optional: You can extract specific keys if needed, e.g.:
                # "point_cloud": lidar_data.get("point_cloud"),
                # "distances": lidar_data.get("distances"),
            }

        return self._state