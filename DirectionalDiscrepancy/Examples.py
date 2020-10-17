from Classes import *
from mpmath import pi
from math import pi as floatpi



# making instance of each class

# Point
print('\nPoint class:')

p1 = Point(0, 1, -1, is_polar=False)      # Cartesian coordinates
p2 = Point(theta=pi/2, phi=0, r=2)        # Polar coordinates
p3 = Point(theta=pi/3, phi=pi/4)          # Default norm is 1
p4 = Point(pi + 2*pi, pi/2 + 4*pi)
p4.adjust_polar_coordinate()              # adjust polar range

print('Cartesian coordinates\t', p1.get_coordinate(), p1.get_polar_coordinate(), p1.norm, sep='\t')
print('Polar coordinates\t\t', p2.get_coordinate(), p2.get_polar_coordinate(), p2.norm, sep='\t')
print('Polar with default norm\t', p3.get_coordinate(), p3.get_polar_coordinate(), p3.norm, sep='\t')
print('adjust polar range\t\t', p4.get_coordinate(), p4.get_polar_coordinate(), p4.norm, sep='\t')


# NormalVector
print('\nNormal_Vector class:')

v1 = NormalVector(0, 6, 8, is_polar=False)
v2 = NormalVector(theta=pi, phi=floatpi/3, r=5, is_polar=True)
v3 = NormalVector(theta=floatpi, phi=floatpi/3, r=5)

print('normalize\t', v1.get_coordinate(), v1.get_polar_coordinate(), v1.norm, sep='\t')
print('norm = 1 in PC', v2.get_coordinate(), v2.get_polar_coordinate(), v2.norm, sep='\t')
print('norm = 1 in PC', v3.get_coordinate(), v3.get_polar_coordinate(), v3.norm, sep='\t')


# PointSet
print('\nPointSet class:')
import os
from termcolor import colored


folderadd = os.getcwd() + "/"


pointset = PointSet(name='test', address=folderadd)



for p in [p1, p2, p3, p4]:
    pointset.add_point(p)

print(colored("Please check test.txt file at", 'green'),
      pointset.point_file_address, colored(" for points' information", 'green'))


pointset2 = PointSet()      # if name and address is not given it makes a folder in the current working directory
pointset2.add_point(p3)     # add_point() lets you to add points one by one

print('name:', pointset2.name, '\t address:', pointset2.point_file_address)

pointset3 = PointSet(points=[p3, p4], address=folderadd + "point_set_3")   # points can be given as a list when the instance is created


print('\n\n projection of pointset on v1 is: ', v1.project(pointset))     # You can get projection of POintSet on thr Normal_Vector v by v.project()

# MAIN

output_folder = folderadd + "/main_test/"

two_points_main = Main(output_address=output_folder, pointset=pointset3)
print(two_points_main.point_set.coordinates_matrix)

polar_output_folder = folderadd + "polar_coordinates/"
polar = Main(output_address=polar_output_folder)
high_phi = polar.make_polar_coordinate(n=20)

print("\n\nnorth pole is local maxima for phi >", high_phi)

pole = NormalVector(0, 0, 1, is_polar=False)
pole_dis, pole_projection = polar.get_discrepancy_data(pole)

print("north pole has discrepancy", pole_dis)


# RUN

# Set max recall

# The higher the below number is, the more effort the algorithm puts to cover problematic directions.  The default value
# for this parameter is 200.
polar.cover_cap_max_recalls = 500


# Runing the algorithm
polar.is_discrepancy_less_than(d=pole_dis, highest_phi=high_phi, last_theta=floatpi)




