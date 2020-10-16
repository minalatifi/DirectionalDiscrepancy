class Point:
    # (x,y,z,False) or (theta,phi,r)

    _log_scale = 0  # This scales with 10^_log_scale for more accurate float calculation
    _precision = 30

    # Below parameters should not change by user
    _scale = 10 ** _log_scale
    _can_change_the_scale = True

    def get_scale():
        return float(Point._scale)

    def change_log_scale(log_scale):
        from mpmath import floor

        if Point._can_change_the_scale:
            Point._log_scale = floor(log_scale)
            Point._scale = 10 ** Point._log_scale
        else:
            print("You can not change the scale since there are already some instances with the previous scale")

    def change_precision(precision):
        if Point._can_change_the_scale:
            Point._precision = precision
        else:
            print("You can not change the precision since there are already some instances with the previous precision")

    def __init__(self, theta, phi, r=None, is_polar=True):
        from mpmath import mpf, mp, norm, fabs, chop

        if not r:
            r = Point._scale
        else:
            r *= Point._scale

        Point._can_change_the_scale = False
        mp.dps = Point._precision

        if not is_polar:

            self.x = mpf(theta)
            self.y = mpf(phi)
            self.z = mpf(r)
            self.norm = norm(self.get_coordinate(), 2)

            self.set_polar_coordinate()

        else:

            self.norm = fabs(r)
            self.theta = theta
            self.phi = phi

            self.set_coordinate()

        # Setting float parameters

        self.floatx = float(self.x)
        self.floaty = float(self.y)
        self.floatz = float(self.z)

    def set_coordinate(self):
        from mpmath import sin, cos

        self.x = self.norm * cos(self.phi) * cos(self.theta)
        self.y = self.norm * cos(self.phi) * sin(self.theta)
        self.z = self.norm * sin(self.phi)

    def set_polar_coordinate(self):
        from mpmath import asin, cos, chop

        self.phi = asin(self.z / self.norm)

        sintheta = chop(self.y / (self.norm * cos(self.phi)))
        theta = asin(sintheta)

        self.theta = theta

    def adjust_polar_coordinate(self):
        #  This function adjusts the polar coordinate (phi, theta)  such that -pi/2 <= phi <= pi/2,
        #  and 0 <= theta < 2 pi, yet we get the same Cartesian coordinate.

        from mpmath import fmod, pi

        theta = fmod(self.theta, 2 * pi)

        modephi = fmod(self.phi, 2 * pi)

        if modephi <= pi / 2:
            self.phi = modephi
            self.theta = theta
        elif modephi <= 3 * pi / 2:
            self.phi = pi - modephi
            self.theta = fmod(theta - pi, 2 * pi)
        else:
            self.phi = modephi - 2 * pi
            self.theta = theta

    def get_polar_coordinate(self):
        return (self.theta, self.phi)

    def get_coordinate(self):
        return (self.x, self.y, self.z)

    def get_info(self):

        ans = self.get_coordinate() + self.get_polar_coordinate() + (self.norm,)
        return ans

    def get_float_coordinate(self):
        from numpy import array

        return array([self.floatx, self.floaty, self.floatz])

    def __str__(self):
        return str(self.get_float_coordinate())


class PointSet:
    from datetime import datetime
    from itertools import count

    create_time = datetime.now()
    _ids = count(1)

    def __init__(self, name=None, address=None, points=[]):
        from os import getcwd

        self.id = next(self._ids)

        if name == None:
            self.name = 'PointSet_' + str(self.id)
        else:
            self.name = name

        if address == None:
            address = getcwd() + '/' + self.create_time.strftime("%Y-%m-%d %H-%M")

        self.point_file_address = address + '/' + self.name + '.txt'

        check_and_make_directory(address)

        # Writing formats
        self.coorspot = "{: <" + str(20) + "}" + "\t"
        self.numspot = "{: <" + str(7) + "}" + "\t"
        self.cut_floats = 12

        self.file = open(self.point_file_address, "w+")
        self.file.write("Point's Coordinates for " + self.name + '\n\n')

        title_list = ['x,', 'y,', 'z,', 'theta,', 'phi,', 'norm']
        title = self.numspot.format("num,")
        title += make_fromate_with_comma(List=title_list, form=self.coorspot)
        title += "\n\n"
        self.file.write(title)
        self.file.close()

        self.coordinates_matrix = []

        self.points = []
        self.size = 0

        for p in points:
            self.add_point(p)

    def add_point(self, point):
        from termcolor import colored

        if self.is_on_surface(point):

            self.size += 1
            self.points.append(point)

            self.file = open(self.point_file_address, "a")

            info = self.get_info_to_write(point)
            self.file.write(info)
            self.file.close()

            self.coordinates_matrix.append(point.get_float_coordinate())
        else:
            print(colored("{} is not on the surface!".format(point.get_float_coordinate()), 'red'))

    def is_on_surface(self, point):
        return point.norm == Point.get_scale()

    def get_info_to_write(self, point):
        to_write = self.numspot.format(str(self.size) + ",")

        info = list(map(float, point.get_info()))
        to_write += make_fromate_with_comma(List=info, form=self.coorspot, cut_num=self.cut_floats)
        to_write += "\n"

        return to_write


class NormalVector:
    # There is no mpmath calculation and accuracy consideration within this class.
    # Uses the usual 10**-15 accuracy of float.
    # Polar coordinate is always adjusted in this class.

    def __init__(self, theta: float, phi: float, r: float = 1, is_polar=True):
        from numpy import linalg
        from termcolor import colored

        def is_acceptable_type(x):

            if isinstance(x, int) or isinstance(x, float):
                return True
            else:
                return False

        if not (is_acceptable_type(theta) and is_acceptable_type(phi) and is_acceptable_type(r)):
            print(colored('Passing not float or int arguments can make the algorithm slow'
                          + "\n(" + str(theta) + ", " + str(phi) + ", " + str(r) + ")", 'red'), type(theta), type(phi),
                  type(r))
            theta = float(theta)
            phi = float(phi)
            r = float(r)

        if not is_polar:

            norm = linalg.norm((theta, phi, r))
            self.x = theta / norm
            self.y = phi / norm
            self.z = r / norm

            self.norm = 1

            self.set_polar_coordinate()

        else:

            self.norm = 1
            self.theta = theta
            self.phi = phi
            self.adjust_polar_coordinate()

            self.set_coordinate()

        self.set_discrepancy_data()

    def set_coordinate(self):
        from math import sin, cos

        self.x = self.norm * cos(self.phi) * cos(self.theta)
        self.y = self.norm * cos(self.phi) * sin(self.theta)
        self.z = self.norm * sin(self.phi)

    def set_polar_coordinate(self):
        from math import asin, cos

        self.phi = asin(self.z / self.norm)

        sintheta = self.y / (self.norm * cos(self.phi))
        self.theta = asin(sintheta)

        self.adjust_polar_coordinate()

    def adjust_polar_coordinate(self):
        #  This function adjusts the polar coordinate (phi, theta)  such that -pi/2 <= phi <= pi/2,
        #  and 0 <= theta < 2 pi, yet we get the same Cartesian coordinate.

        from math import fmod, pi

        theta = fmod(self.theta, 2 * pi)

        modephi = fmod(self.phi, 2 * pi)

        if modephi <= pi / 2:
            self.phi = modephi
            self.theta = theta
        elif modephi <= 3 * pi / 2:
            self.phi = pi - modephi
            self.theta = fmod(theta - pi, 2 * pi)
        else:
            self.phi = modephi - 2 * pi
            self.theta = theta

    def get_coordinate(self):
        from numpy import array
        return array([self.x, self.y, self.z])

    def get_polar_coordinate(self):
        from numpy import array
        return array([self.theta, self.phi])

    def get_info(self):
        from numpy import append

        # print("normal vector: get_info ", self.get_coordinate(), self.get_polar_coordinate(), self.norm)
        ans = append(self.get_coordinate(), self.get_polar_coordinate())
        ans = append(ans, self.norm)
        return ans

    def project(self, pointset):
        # treat points as they are on the unit ball
        from numpy import matmul

        ans = matmul(pointset.coordinates_matrix, self.get_coordinate())

        return ans

    def set_discrepancy_data(self, discrepancy=None, confidence_radius=None, expected_radius=None,
                             covering_type="None", index=None):

        self.directional_discrepancy = discrepancy
        self.confidence_radius = confidence_radius
        self.expected_radius = expected_radius
        self.covering_type = covering_type
        self.index = index

        if (confidence_radius is not None) and (expected_radius is not None):
            self.covered_radius = max(confidence_radius, expected_radius)
        elif confidence_radius is not None:
            self.covered_radius = confidence_radius
        else:
            self.covered_radius = None

    def rotate(self, rotation_matrix):
        from numpy import matmul

        self.x, self.y, self.z = matmul(rotation_matrix, self.get_coordinate())

        self.set_polar_coordinate()
        self.set_discrepancy_data()

    def __str__(self):
        return str(self.get_coordinate())

    def write_comma_separated(self, address=None):
        from os import getcwd

        if address is None:
            add = getcwd() + "/normal_vectors.txt"
        else:
            add = address

        towrite = ", ".join(map(str, self.get_coordinate())) + ",\n"

        file = open(add, "a")
        file.write(towrite)
        file.close()


class Main:
    def __init__(self, output_address=None, pointset=None):
        from os import getcwd
        from datetime import datetime as dt

        if output_address == None:
            add = getcwd()
            now = dt.now()
            add += "/Discrepancy_output_"
            check_and_make_directory(add)
            add += "on_" + str(now.date()) + "_at_" + str(now.hour) + "_" + str(now.minute) + "/"
            check_and_make_directory(add)

            self.output_folder = add
        else:
            check_and_make_directory(output_address)
            self.output_folder = output_address

        self.make_file_addresse()

        if pointset == None:
            self.point_set = PointSet(address=self.output_folder)
            self.min_discrepancy_distance = None
        else:
            self.point_set = pointset
            self.min_discrepancy_distance = 1 / self.point_set.size

        self.points_address = self.point_set.point_file_address
        self.max_dis_found = -1
        self.max_dis_direction_found = None

        # Covering parameters
        self.directions = []
        self.cap_covering_number = 7
        self.cover_cap_max_recalls = 200
        self.cover_cap_shift_num = 1
        self.reset_discrepancy_parameters()
        self.calls_num = 0
        self.calls_list = []

        # Below forms will be used to format the output textfiles
        self.numspot = "{: <" + str(7) + "}" + "\t"
        self.coorspot = "{: <" + str(20) + "}" + "\t"
        self.cut_floats = 12

    def make_file_addresse(self):
        # Making file addresses
        self.directions_file_address = self.output_folder + "directions.txt"
        self.cover_cap_file_address = self.output_folder + "cover_cap.txt"
        self.orbits_file_address = self.output_folder + "orbits.txt"

    def reset_output_address(self, new_address):

        self.output_folder = new_address
        check_and_make_directory(self.output_folder)

        self.make_file_addresse()

    def make_polar_coordinate(self, n, is_twisted=True, precision=30, log_scale=0):
        # This function distribute points with polar coordinate on sphere.  This algorithm is O (n^2).
        # It also returns a phi above which we know north pole is a local maximum.

        from mpmath import mp, sin, cos, sqrt, pi, floor, mpf, asin
        from numpy import array, cross
        from numpy.linalg import norm
        from os import remove

        mp.dps = precision

        remove(self.points_address)
        if is_twisted:
            name = "twisted_polar_coordinates_" + str(n)
            self.point_set = PointSet(name=name, address=self.output_folder)
        else:
            name = "polar_coordinates_" + str(n)
            self.point_set = PointSet(name=name, address=self.output_folder)

        self.points_address = self.point_set.point_file_address

        if not float(Point._log_scale) == log_scale:
            Point.change_log_scale(log_scale)
            print("Point scale is", Point.get_scale())

        if not Point._precision == precision:
            Point.change_precision(precision)

        phis = [0] * (n - 1)
        m = [0] * (n - 1)

        for j in range(n - 1):
            phis[j] = ((pi * (j + 1)) / n) - (pi / 2)
            m[j] = int(floor(mpf(".5") + sqrt(3) * n * cos(phis[j])))

        for j in range(n - 1):
            for i in range(m[j]):

                if is_twisted:
                    if j + 1 < n / 2:
                        shift = (2 * pi * (j + 1)) / (n * m[j])
                    else:
                        shift = (2 * pi * (1 - mpf(j + 1) / mpf(n))) / m[j]
                else:
                    shift = 0

                theta = ((2 * pi * i) / m[j]) + shift

                new_point = Point(theta=theta, phi=phis[j])
                self.point_set.add_point(new_point)

        north_pole = Point(theta=0, phi=pi / 2)
        south_pole = Point(theta=0, phi=-pi / 2)
        self.point_set.add_point(north_pole)
        self.point_set.add_point(south_pole)

        self.min_discrepancy_distance = 1 / self.point_set.size

        # In below we get the confidence phi around the north pole

        highest_phi = - 1
        for i in range(int(n / 2) - 1, n - 2):
            phi1 = phis[i]
            phi2 = phis[i + 1]

            v1 = array([mpf(0), cos(phi1), mpf(0)])
            v2 = array([2 * cos(phi1), mpf(0), sin(phi2) - sin(phi1)])

            normal_vector = cross(v1, v2)
            normal_vector /= norm(normal_vector)

            highest_phi = max(highest_phi, asin(abs(normal_vector[2])))

        return float(highest_phi)

    def get_discrepancy_data(self, direction: NormalVector):

        projection = direction.project(self.point_set)
        discrepancy = self.get_directional_discrepancy(projection)

        if discrepancy > self.max_dis_found:
            self.max_dis_direction_found = direction
            self.max_dis_found = discrepancy

        return discrepancy, projection

    def get_confidence_radius(self, d, discrepancy, projection):
        # It returns cr with this property for all directions in close disk B(v, cr) like u we have disu <= dis.
        # It is O (|projection|) = O (N).
        from math import floor

        if discrepancy >= d - self.min_discrepancy_distance:
            print(" There is a problem in passed direction to the get_confidence_radius ")
            return 0

        n = self.point_set.size
        available_change = floor(n * (d - discrepancy)) + 1
        cr = self.max_interval(projection, available_change)

        cr /= Point.get_scale()

        return cr

    def is_discrepancy_less_than(self, d, highest_phi=None, lowest_phi=0, first_theta=0, last_theta=None,
                                 dtheta_method='dynamic'):
        from math import pi, asin
        from time import time

        start = time()

        if highest_phi:
            self.highest_phi = highest_phi
        else:
            self.highest_phi = pi / 2

        if last_theta:
            self.last_theta = last_theta
        else:
            self.last_theta = 2 * pi

        self.lowest_phi = lowest_phi
        self.first_theta = first_theta

        phi_check = 0 <= self.lowest_phi and self.lowest_phi < self.highest_phi and self.highest_phi <= pi / 2

        if not phi_check:
            print("There is an issue in phi constrains")
            return False

        theta_check = 0 <= self.first_theta and self.first_theta < self.last_theta and self.last_theta <= 2 * pi

        if not theta_check:
            print("There is an issue in theta constrains")
            return False

        self.reset_discrepancy_parameters()
        self.is_complete_orbit = (self.first_theta == 0) and (self.last_theta == 2 * pi)
        self.checked_discrepancy_upper_bound = d

        # making directions file
        title = ["x", "y", "z", "theta", "phi", "norm", "discrepancy", "conf. r", "covered radius", "covering type"]
        self.write_title(title_list=title, file_address=self.directions_file_address)

        # making cover cap file
        title = []
        self.write_title(title_list=title, file_address=self.cover_cap_file_address)

        # making orbit file
        title = ["phi", "expected radius", "dtheta", "covered phi", "fd", "ld", "#d"]
        self.write_title(title_list=title, file_address=self.orbits_file_address, is_orbit_file=True)

        if self.highest_phi == pi / 2:
            could_add = self.add_direction(theta=0, phi=pi / 2, d=d, expected_radius=0)
            # We can put a positive expected radius

            if not could_add:
                print("The algorithm cannot handle the north pole!")
                return False
            else:
                confidence_r = self.directions[0].confidence_radius
                covered_phi = asin(r_to_dot(confidence_r))
        else:
            covered_phi = self.highest_phi

        while covered_phi > self.lowest_phi:

            phi, expected_radius, mindtheta = self.get_phi_expected_r_dtheta(d=d, covered_phi=covered_phi)
            # if expected_radius is 0 we should stop, or do something

            if expected_radius == 0:
                self.write_report(start=start, is_covered=False)
                return False

            theta = self.first_theta
            minind = self.direction_number + 1
            should_continue = True

            while should_continue:

                could_add = self.add_direction(theta=theta, phi=phi, d=d, expected_radius=expected_radius)
                if not could_add:
                    last_direction, discrepancy, projection = self.create_direction(theta=theta, phi=phi)

                    cr = self.get_confidence_radius(d=d, discrepancy=discrepancy, projection=projection)

                    last_direction.directional_discrepancy = discrepancy
                    last_direction.confidence_radius = cr

                    self.write_report(start=start, is_covered=False, found_higher_disrepancy=True)
                    return False

                direction = self.directions[-1]
                dtheta = self.get_dtheta(direction=direction, mindtheta=mindtheta, dtheta_method=dtheta_method)

                # The below lets the cover go one direction after last_theta if the orbit is not complete
                if not self.is_complete_orbit:
                    should_continue = theta < self.last_theta

                theta += dtheta

                if self.is_complete_orbit:
                    should_continue = theta < self.last_theta

            self.orbits_number += 1
            maxind = self.direction_number
            covered_phi = self.get_covered_phi(minind=minind, maxind=maxind, phi=phi,
                                               dtheta=dtheta, dtheta_method=dtheta_method)
            self.write_orbit_data(phi=phi, expected_r=expected_radius, dtheta=dtheta, minind=minind, maxind=maxind,
                                  coveredphi=covered_phi)

        cover_cap_index_file_add = self.output_folder + "cover_cap_indices.txt"


        first_round_run_time = time() - start

        cover_cap_start = time()
        answer = True
        not_covered_with_cover_cap = []
        for ind in self.not_covered_directions:
            direction = self.directions[ind]
            could_cover, found_higher_discrepancy, calls_num = self.cover_cap(direction)

            if found_higher_discrepancy:
                self.write_report(start=start, is_covered=False, found_higher_disrepancy=True)
                return False
            elif not could_cover:
                answer = False
                not_covered_with_cover_cap += [ind]
            self.calls_list.append(calls_num)

        cover_cap_run_time = time() - cover_cap_start
        cover_cap_ind = open(cover_cap_index_file_add, mode="w+", buffering=1)
        cover_cap_ind.write("number\t" + "direction_ind\t" + "# of covercap\t\n")
        for i in range(len(self.not_covered_directions)):
            cover_cap_ind.write(str(i) + "\t" + str(self.not_covered_directions[i]) + "\t" + str(self.calls_list[i]) + "\t\n")
        cover_cap_ind.close()

        if not answer:
            issue_file_add = self.output_folder + "not_covered_directions.txt"
            file = open(issue_file_add, "w+")
            file.write("The following directions could not be covered\n")
            file.close()
            write_comma_separated(List=not_covered_with_cover_cap, address=issue_file_add)

        self.write_report(start=start, is_covered=answer, problematic_directions=not_covered_with_cover_cap,
                          first_round_run_time=first_round_run_time, cover_cap_run_time=cover_cap_run_time)
        return answer

    def cover_cap(self, direction: NormalVector):

        self.cover_cap_calls += 1

        calls_num = 0
        found_higher = False
        could_cover = False

        file = open(self.cover_cap_file_address, "a")
        to_write = "\n\nDirection number " + str(direction.index) + "\n"
        file.write(to_write)
        file.close()

        cannot_cover = [direction]

        while calls_num <= self.cover_cap_max_recalls:
            calls_num += 1

            if cannot_cover == []:
                could_cover = True
                return could_cover, found_higher, calls_num
            else:
                cover_direction = cannot_cover.pop(0)

            self.write_cover_cap_goal(cover_direction)
            result, outcome, candidates = self.cover_cap_choose_best_shift(direction=cover_direction)

            if result == "found higher discrepancy at":
                direction.covering_type = "not covered"
                found_higher = True
                return could_cover, found_higher, calls_num
            elif result == "all covered by":
                self.write_cover_cap(direction=outcome)
                continue
            else:
                for direction in candidates:
                    self.write_cover_cap(direction=direction)
                cannot_cover += outcome

        if not could_cover:
            direction.covering_type = "not covered"

        return could_cover, found_higher, calls_num

    # functions that are used inside a callable function.  These are not supposed to be called by the user.

    # constructive functions

    def get_directional_discrepancy(self, projections):
        # It takes a set of projection of points on a direction.Then returns the  discrepancy of this direction.
        # This algorithm is O (| points |) = O (n)

        scale_factor = Point.get_scale()

        sorted_proj = sorted(projections, reverse=True)
        size = self.point_set.size

        positives = 0
        zeros = 0
        answer = -1
        ind = 0

        while ind < size:

            positives += zeros

            current_proj = sorted_proj[ind]
            zeros = 1
            while ind < size - 1 and sorted_proj[ind + 1] == current_proj:
                zeros += 1
                ind += 1

            cap_area = (scale_factor - current_proj) / 2
            cap_area /= scale_factor
            # print("cap area =", cap_area, " pos = ", positives, " zero = ", zeros)
            answer = max(answer, abs((positives / size) - cap_area), abs(((positives + zeros) / size) - cap_area))
            # print("answer = ", answer)

            ind += 1

        return answer

    def create_direction(self, theta, phi):

        direction = NormalVector(theta, phi)
        discrepancy, projection = self.get_discrepancy_data(direction)

        return direction, discrepancy, projection

    def add_direction(self, theta, phi, d, expected_radius):

        direction, discrepancy, projection = self.create_direction(theta=theta, phi=phi)
        direction.directional_discrepancy = discrepancy

        if discrepancy >= d - self.min_discrepancy_distance:
            print("Discrepancy is " + str(discrepancy)[:10] + " at the following direction!")
            print(str(direction.get_coordinate()))
            return False
        else:
            confidence_r = self.get_confidence_radius(d=d, discrepancy=discrepancy, projection=projection)

            if confidence_r < expected_radius:
                covering_type = "cover cap"
            else:
                covering_type = "Normal"

            self.direction_number += 1
            direction.set_discrepancy_data(discrepancy=discrepancy, confidence_radius=confidence_r,
                                           expected_radius=expected_radius, covering_type=covering_type,
                                           index=self.direction_number)

            if direction.covering_type == "cover cap":
                self.not_covered_directions += [direction.index]

            self.directions += [direction]
            self.write_direction(direction=direction)

            return True

    def reset_discrepancy_parameters(self):

        self.checked_discrepancy_upper_bound = 0
        self.direction_number = -1
        self.directions = []
        self.orbits_number = 0
        self.cover_cap_calls = 0
        self.not_covered_directions = []
        self.is_complete_orbit = None

    def which_cannot_cover_with_half_radius(self, v: NormalVector, shift=0):
        from math import asin, pi

        d = self.checked_discrepancy_upper_bound
        r = v.expected_radius
        half_r = r / 2

        distance_to_center = .86 * r
        larger_r = distance_to_center + r

        phi_arround_pole = asin(r_to_dot(distance_to_center))

        dtheta = 2 * pi / self.cap_covering_number
        candidates = [NormalVector(theta=0, phi=pi / 2)]
        candidates += [NormalVector(phi=phi_arround_pole, theta=shift + (i * dtheta))
                       for i in range(self.cap_covering_number)]

        rotation = get_rotation_matrix_north_pole_to(theta=v.theta, phi=v.phi)

        cannot_cover = []
        for cand_v in candidates:
            cand_v.rotate(rotation)

            discrepancy, projection = self.get_discrepancy_data(cand_v)

            if discrepancy >= d:
                return "found higher discrepancy at", cand_v, candidates
            else:
                cr = self.get_confidence_radius(d=d, discrepancy=discrepancy, projection=projection)
                cand_v.set_discrepancy_data(discrepancy=discrepancy, confidence_radius=cr, expected_radius=half_r)

                if cr >= larger_r:
                    return "all covered by", cand_v, candidates
                elif cr < half_r:
                    cannot_cover += [cand_v]

        return "cannot list", cannot_cover, candidates

    def cover_cap_choose_best_shift(self, direction: NormalVector):
        from math import pi

        all_outcomes = []
        for i in range(self.cover_cap_shift_num):

            shift = (i * 2 * pi) / (self.cover_cap_shift_num * self.cap_covering_number)

            result, output, candidates = self.which_cannot_cover_with_half_radius(v=direction, shift=shift)

            if result == "found higher discrepancy at" or result == "all covered by":
                return result, output, candidates
            else:
                all_outcomes += [[len(output), result, output, candidates]]

        all_outcomes.sort(key=lambda l: l[0])
        not_cover_size, result, outcome, candidates = all_outcomes[0]

        return result, outcome, candidates

    def max_interval(self, collection: list, k):
        # collection is a repeated subset of [-r, r], where r is the scale factor of class Point. It returns biggest l
        # so that any subinterval of [-r, r] with length less than l intersects collection in less than k points.
        # r is the radius of the sphere at which the points are located. This algorithm is O (| collection |).

        n = self.point_set.size
        ans = 2 * Point.get_scale()

        if n < k:
            return ans

        sorted_set = sorted(collection)
        for i in range(n - k + 1):
            ans = min(ans, sorted_set[k + i - 1] - sorted_set[i])

        return ans

    def get_dtheta_at(self, phi, covered_r):
        from math import asin, cos, sqrt, pi

        if covered_r >= sqrt(2) * cos(phi):
            # In this case, the confidence radius is so large such that it covers the whole orbit and the north pole.
            dtheta = pi
        else:
            dtheta = 2 * asin((.5 * covered_r) / cos(phi))
        return dtheta

    def get_dtheta(self, direction: NormalVector, mindtheta, dtheta_method):

        if dtheta_method == 'dynamic':
            return self.get_dtheta_at(phi=direction.phi, covered_r=direction.covered_radius)
        elif dtheta_method == 'constant':
            return mindtheta

    def get_phi_expected_r_dtheta(self, d, covered_phi):
        #   It returns two numbers phi, expected_radius with this property: in orbit Sin (phi) two sphere with radius expected_radius has
        #  upper intersection upper than covered_latitude. It is O (n).
        # Basically it runs a binary search to find an optimal phi for the next level.
        from termcolor import colored

        if covered_phi <= 0:
            print("get_phi_expected_r_dtheta: ", colored("The semi-sphere is already covered", 'red'))
            return None, None, None

        expected_r = self.approximate_confidence_raduis(phi=covered_phi, d=d)

        if expected_r == 0:
            print("get_phi_expected_r_dtheta: ", colored("The algorithm is going to stuck since expected_r = 0", 'red'))
            return covered_phi, 0, 0

        phi = go_down(covered_phi, 2 * expected_r)

        up = covered_phi
        down = phi
        last_accepted = up
        last_er = expected_r

        # This loop does a binary search with 10 steps to find the best phi
        for i in range(10):
            phi = .5 * (up + down)
            expected_r = self.approximate_confidence_raduis(phi, d)
            dtheta = self.get_dtheta_at(phi=phi, covered_r=expected_r)
            v1 = NormalVector(theta=0, phi=phi)
            v2 = NormalVector(theta=dtheta, phi=phi)

            has_intersection, up_ans, down_ans = self.has_intersection(v1, expected_r, v2, expected_r)

            if has_intersection:
                if up_ans.phi >= covered_phi:
                    up = phi
                    last_accepted = phi
                    last_er = expected_r

                else:
                    down = phi

            else:
                print("There is a problem in get_phi_expected_r_dtheta")
                print(v1, v2, " do not intersect with radius ", expected_r)

        if last_accepted <= 0:
            phi = 0
            expected_r = self.approximate_confidence_raduis(phi, d)
        else:
            phi = last_accepted
            expected_r = last_er

        last_dtheta = self.get_dtheta_at(phi=phi, covered_r=expected_r)

        return phi, expected_r, last_dtheta

    def get_covered_phi(self, minind, maxind, phi, dtheta, dtheta_method):

        from math import sin, cos, atan
        from termcolor import colored

        #   in this block we find the minimum confidence radius in the given orbit
        if dtheta_method == 'constant':
            r = 1
            for i in range(minind, maxind + 1):
                ri = self.directions[i].covered_radius
                r = min(r, ri)

            dot = r_to_dot(r)
            tantheta = (1 - cos(dtheta)) / sin(dtheta)
            covered_theta = atan(tantheta)

            a = cos(phi) * cos(covered_theta)
            b = sin(phi)

            up, down = solve_acos_bsin(a, b, dot)
            answer = down
        elif dtheta_method == 'dynamic':
            answer = - 2
            for i in range(minind, maxind):
                dir1 = self.directions[i]
                dir2 = self.directions[i + 1]
                # print("inds, dirs ", i, i + 1, dir1, dir2)
                has_intersection, up, down = self.has_intersection(v1=dir1, r1=dir1.covered_radius,
                                                                   v2=dir2, r2=dir2.covered_radius)
                # print("after has intersection ", i, i + 1, dir1, dir2)
                if not has_intersection:
                    if self.are_nested(v1=dir1, r1=dir1.covered_radius, v2=dir2, r2=dir2.covered_radius):
                        continue
                    else:
                        print(
                            colored("get_covered_phi: There is an error in directions " + str(i) + " and " + str(i + 1),
                                    "red"))
                    # print("after are nested ", i, i + 1, dir1, dir2)

                else:
                    answer = max(answer, down.phi)

            if self.is_complete_orbit:

                dir1 = self.directions[minind]
                dir2 = self.directions[maxind]

                has_intersection, up, down = self.has_intersection(v1=dir1, r1=dir1.covered_radius,
                                                                   v2=dir2, r2=dir2.covered_radius)
                if not has_intersection:
                    if not self.are_nested(v1=dir1, r1=dir1.covered_radius, v2=dir2, r2=dir2.covered_radius):
                        print(colored("get_covered_phi: There is an error in directions "
                                      + str(minind) + " and " + str(maxind), "red"))
                        print(dir1, dir2)
                else:
                    answer = max(answer, down.phi)

        return answer

    def approximate_confidence_raduis(self, phi, d):
        # It returns minimum expected confidence radius in orbit with height sin(Phi]).
        #  This algorithm is O(|points|= O(n))
        test_num = 20
        dtheta = (self.last_theta - self.first_theta) / test_num
        theta = self.first_theta

        radius_set = []
        for i in range(test_num):

            direction, discrepancy, projection = self.create_direction(theta=theta, phi=phi)

            if discrepancy >= d:
                return 0

            confidence_r = self.get_confidence_radius(d=d, discrepancy=discrepancy, projection=projection)
            radius_set += [confidence_r]

            theta += dtheta

        def get_min_radius(set):
            # This function implement the criteria with which we want to choose our minimum out of the sample.
            from math import floor
            from statistics import median

            sorted_set = sorted(set)
            min_acceptable = .5 * median(sorted_set)

            ind = 0
            while sorted_set[ind] < min_acceptable:
                ind += 1

            ans = sorted_set[ind]

            return ans

        ar = get_min_radius(radius_set)

        return ar

    def has_intersection(self, v1: NormalVector, r1: float, v2: NormalVector, r2: float):
        #   v1 and v2 are two distinict points on unit upper semisphere.It returns true if two balls B[v1, r1] and
        # B[v2, r2] has intersection on unit sphere and in this case answers will be returned as
        # up_intersect and down_intersect relative to Z - axis. It is O(1)

        from numpy import cross, dot
        from numpy.linalg import norm
        from math import sqrt

        t1 = r_to_dot(r1)
        t2 = r_to_dot(r2)

        u1 = v1.get_coordinate()
        u2 = v2.get_coordinate()
        t = dot(u1, u2)

        dot_equation_solution_on_u1u2 = ((t1 - t * t2) / (1 - t ** 2)) * u1 + ((t2 - t * t1) / (1 - t ** 2)) * u2
        norm_on_u1u2 = norm(dot_equation_solution_on_u1u2)
        if norm_on_u1u2 > 1:
            return False, None, None

        perp = cross(u1, u2)
        perp /= norm(perp)

        if perp[2] < 0:
            perp *= -1

        perp_projection = sqrt(1 - norm_on_u1u2 ** 2) * perp

        up = dot_equation_solution_on_u1u2 + perp_projection
        down = dot_equation_solution_on_u1u2 - perp_projection

        up_intersection = NormalVector(up[0], up[1], up[2], is_polar=False)
        down_intersection = NormalVector(down[0], down[1], down[2], is_polar=False)

        return True, up_intersection, down_intersection

    def are_nested(self, v1: NormalVector, r1: float, v2: NormalVector, r2: float):
        from numpy.linalg import norm

        distance = norm(v1.get_coordinate() - v2.get_coordinate())

        return max(r1, r2) >= distance + min(r1, r2)

    # Writing functions

    def write_title(self, title_list: list, file_address, is_orbit_file=False):

        file = open(file_address, "w+")
        file.write("This run aims to check if discrepancy is less than "
                   + str(self.checked_discrepancy_upper_bound)[:6] + "\n")
        if is_orbit_file:
            file.write("each row contains covering data for one orbit at phi, i.e. z = sin(phi), with the range "
                       "of indices of directions corresponding to this orbit and the expected confidence radius at it."
                       " dtheta shows the amount that directions vary horizontally at this orbit." +
                       "\n fd = first direction, ld = last direction, #d = number of directions " + "\n\n")
            title = self.numspot.format("number,") + make_fromate_with_comma(List=title_list[:-3], form=self.coorspot) \
                    + make_fromate_with_comma(List=title_list[-3:], form=self.numspot) + "\n\n"
        else:

            title = self.numspot.format("number,") + make_fromate_with_comma(List=title_list,
                                                                             form=self.coorspot) + "\n\n"
        file.write(title)
        file.close()

    def write_direction(self, direction: NormalVector):

        towritelist = list(direction.get_info()) + [direction.directional_discrepancy, direction.confidence_radius,
                                                    direction.covered_radius, direction.covering_type]
        towriteformated = make_fromate_with_comma(towritelist, form=self.coorspot, cut_num=self.cut_floats)

        to_write = self.numspot.format(str(self.direction_number) + ",") + towriteformated + "\n"

        file = open(self.directions_file_address, "a")
        file.write(to_write)
        file.close()

    def write_orbit_data(self, phi, expected_r, dtheta, minind, maxind, coveredphi):

        towritelist = [phi, expected_r, dtheta, coveredphi]
        indlist = [minind, maxind, maxind - minind + 1]
        towriteformated = make_fromate_with_comma(towritelist, form=self.coorspot, cut_num=self.cut_floats) \
                          + make_fromate_with_comma(indlist, form=self.numspot)

        to_write = self.numspot.format(str(self.orbits_number) + ",") + towriteformated + "\n"

        file = open(self.orbits_file_address, "a")
        file.write(to_write)
        file.close()

    def write_cover_cap_goal(self, direction: NormalVector):

        to_write = "\n Goal is to cover a ball around direction " + direction.__str__() + " with radius " \
                   + str(direction.expected_radius) + ".\n We do so with balls with half radius = " \
                   + str(direction.expected_radius / 2) + "\n"

        header_list = ["x", "y", "z", "confidence radius", "expected radius", "is covered?", "directional discrepancy"]
        header = make_fromate_with_comma(header_list, form=self.coorspot) + "\n"

        file = open(self.cover_cap_file_address, "a")
        file.write(to_write)
        file.write(header)
        file.close()

    def write_cover_cap(self, direction: NormalVector):

        towritelist = list(direction.get_coordinate()) + [direction.confidence_radius, direction.expected_radius,
                                                          direction.confidence_radius >= direction.expected_radius,
                                                          direction.directional_discrepancy]
        towriteformated = make_fromate_with_comma(towritelist, form=self.coorspot, cut_num=self.cut_floats)
        to_write = towriteformated + "\n"

        file = open(self.cover_cap_file_address, "a")
        file.write(to_write)
        file.close()

    def write_report(self, start, is_covered=True, found_higher_disrepancy=False, problematic_directions=[],
                     first_round_run_time=0, cover_cap_run_time=0):
        from time import time, ctime

        end = time()

        towrite = "This run aimed to show the discrepancy is less than " + str(self.checked_discrepancy_upper_bound) \
                  + "\nin the range " + str(self.lowest_phi) + " <= phi <= " + str(self.highest_phi) + " and \n" \
                  + str(self.first_theta) + " <= theta <= " + str(self.last_theta) \
                  + ".\n"
        towrite += "There were " + str(self.point_set.size) + " points on the sphere.\n"
        if found_higher_disrepancy:
            towrite += "It found a direction with discrepancy higher than " + str(self.checked_discrepancy_upper_bound) + " - 1/" \
                       + str(self.point_set.size) + ". \n"
        elif is_covered:
            towrite += "It has successfully proved the upper bound. \n"
        else:
            towrite += "The algorithm had issue to cover " + str(len(problematic_directions)) + \
                       " directions.  Please check the file not_covered.txt for details.\n"

        towrite += "It starts at " + ctime(start) + " and ends at " + ctime(end) + ".  Run time is " \
                   + str(round((end - start) / 60, 2)) + " minutes. The first round run time is " \
                   + str(round(first_round_run_time / 60, 2)) + " minutes, and the cover cap run time is " \
                   + str(round(cover_cap_run_time / 60, 2)) + " minutes.\n"
        towrite += "It has covered the semi-sphere with " + str(self.direction_number + 1) + " directions in " \
                   + str(self.orbits_number) + " orbits.\n"
        towrite += "The max directional discrepancy found is " + str(self.max_dis_found) + \
                   " which happens at direction " + str(self.max_dis_direction_found) + "\n"
        towrite += str(len(self.not_covered_directions)) + " directions needed to be cover by cover cap.\n"

        towrite += "Points where on a Sphere with radius " + str(Point.get_scale()) + ".\n"

        print("", towrite, sep="\n")
        report_add = self.output_folder + "report.txt"
        report_file = open(report_add, "w+")
        report_file.write(towrite)
        report_file.close()


# Useful tools

def check_and_make_directory(address):
    from os import path, mkdir
    from termcolor import colored

    if not path.exists(address):
        mkdir(address)

def r_to_dot(r):
    scale = 1  # Decide what we are covering, the unit ball or scaled one.
    dot = (2 * (scale ** 2) - r ** 2) / 2;

    return dot


def go_down(phi, r):
    # It returns the orbit new_phi which is below phi and has distance r
    #  (on the sphere on which our directions are located)
    from math import asin, sin, pi

    if sin(phi) - r >= -1:
        new_phi = asin(sin(phi) - r)
    else:
        new_phi = - pi / 2

    return new_phi


def solve_acos_bsin(a, b, c):
    #   It solves the equation  $ a cos(theta) + b sin(theta) = c $ for theta as the unknown.
    #   This has either two or no answer.  It is needed for the get_covered_phi function.
    #   It always returns the larger answer first.
    from numpy.linalg import norm
    from math import acos

    Norm = norm((a, b))
    if abs(c) > Norm:
        return None, None

    dot = c / Norm
    dtheta = acos(dot)

    if b >= 0:
        angle = acos(a / Norm)
    else:
        angle = - acos(a / Norm)

    upans = angle + dtheta
    downans = angle - dtheta

    return upans, downans


def write_comma_separated(List: list, address=None):
    from os import getcwd

    if address is None:
        add = getcwd() + "/list.txt"
    else:
        add = address

    file = open(add, "a+")
    for l in List:
        if type(l) is list or type(l) is tuple:
            towrite = ", ".join(map(str, l)) + ",\n"
        else:
            towrite = str(l) + ",\n"
        file.write(towrite)

    file.close()


def make_fromate_with_comma(List: list, form: str, cut_num=10):
    ans = ""
    for l in List:
        if isinstance(l, float):
            cut_l = round(l, cut_num)
        else:
            cut_l = l
        ans += form.format(str(cut_l) + ",")

    return ans


def get_rotation_matrix_north_pole_to(theta, phi):
    from numpy import array, matmul
    from math import sin, cos

    Rx = array([[1, 0, 0], [0, sin(phi), cos(phi)], [0, -cos(phi), sin(phi)]])
    Rz = array([[sin(theta), cos(theta), 0], [-cos(theta), sin(theta), 0], [0, 0, 1]])
    rotation_matrix = matmul(Rz, Rx)

    return rotation_matrix