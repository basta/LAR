from robolab_turtlebot import Turtlebot, Rate

turtle = Turtlebot()
rate = Rate(10)

while True:
    i = input()
    if i == 'w':
        turtle.cmd_velocity(linear=0.1)
    elif i == 's':
        turtle.cmd_velocity(linear=-0.1)
    elif i == 'a':
        turtle.cmd_velocity(angular=0.1)
    elif i == 'd':
        turtle.cmd_velocity(angular=-0.1)

    print(turtle.get_odometry())

    rate.sleep()
