# Single tracked obstacle
std_msgs/Header header
geometry_msgs/Point position      # x, y in robot's odom frame
geometry_msgs/Vector3 velocity    # estimated dx/dt, dy/dt (m/s)
float32 radius                    # estimated bounding radius (m)
int32 id                          # persistent tracking ID
