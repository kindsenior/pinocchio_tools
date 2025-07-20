cmake_minimum_required(VERSION 3.0.2)
project(pinocchio_tools)

find_package(catkin REQUIRED)

# Python modules
catkin_python_setup()

catkin_package()

# Install Python libraries for releasse
# install(
#   DIRECTORY src/pinocchio_tools
#   DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
# )

# python scripts
# catkin_install_python(
#   PROGRAMS
#     scripts/run_example.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
