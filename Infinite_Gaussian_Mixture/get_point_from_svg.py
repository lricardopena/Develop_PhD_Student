import xml.etree.ElementTree as ET


ELLIPSE_TAG_NAME = '{http://www.w3.org/2000/svg}ellipse'
GROUP_TAG_NAME = '{http://www.w3.org/2000/svg}g'

class get_points_by_svg:

    def __init__(self, filename):
        self.svg_tree = self.__read_svg_file(filename)

    def get_group_by_id(self, group_id):
        return [circle
                for group in self.svg_tree.iter(GROUP_TAG_NAME)
                if 'id' in group.attrib
                if group.attrib['id'] == group_id
                for circle in self.__get_all_points(group)]
    @staticmethod
    def __circle_to_point(circle):
        return (float(circle.attrib['cx']), float(circle.attrib['cy']))

    @staticmethod
    def __read_svg_file(svg_file_name):
        return ET.parse(svg_file_name)

    def get_all_points(self):
        return [self.__circle_to_point(circle) for circle in self.svg_tree.iter(ELLIPSE_TAG_NAME)]
        #search the tag and see the name of the tag

    def __get_all_points(self, tree):
        return [self.__circle_to_point(circle) for circle in tree.iter(ELLIPSE_TAG_NAME)]
        #search the tag and see the name of the tag

    def get_point_by_id(self, tree, point_id):
        return [self.__circle_to_point(circle)
                for circle in tree.iter(ELLIPSE_TAG_NAME)
                if 'id' in circle.attrib
                if circle.attrib['id'] == point_id]




#pivot = get_point_by_id(svg_tree, 'pivot')
#points = get_group_by_id(svg_tree, 'class1')

#print(pivot)
#print (points)