from .hello_robot import HelloRobot
from ..rpc import RPCServer
import atexit


class Listener:
    def __init__(self, robot, robot_config, port_configs, *args, **kwargs):
        self.robot = robot
        print("starting robot listener")
        if self.robot is None:
            if robot_config is not None:
                self.robot = HelloRobot(**robot_config)
            else:
                self.robot = HelloRobot()

        self.robot.home()
        self.server = RPCServer(self.robot, port_configs["host"], port_configs["action_port"])
    
    def stream(self):
        self.server.start()
        print("server started")
    
    @atexit.register
    def stop(self):
        self.server.stop()
        print("server stopped")
    