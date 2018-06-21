""""""
from typing import Dict, List, NamedTuple

import numpy as np

from active_shape_model import ActiveShapeModel
from image_shape import ImageShape
from incisors import Incisors
from point import Point

# Types
InitCoord = NamedTuple("InitCoord", [("bottom_left", Point), ("top_right", Point)])
AsmDict = Dict[Incisors, ActiveShapeModel]
ImageShapesDict = Dict[Incisors, List[ImageShape]]
InitCoordsDict = Dict[Incisors, List[InitCoord]]


class Initializator:
    def initialize(
        self, asm_dict: AsmDict, images: List[np.ndarray]
    ) -> ImageShapesDict:
        raise NotImplementedError


class ManualInitializator(Initializator):
    """Manual initializator for incisor positions in images (hardcoded)"""

    IMAGE_COORDS = {
        Incisors.UPPER_OUTER_LEFT: [
            InitCoord(Point(1303, 733), Point(1417, 1005)),
            InitCoord(Point(1312, 652), Point(1391, 963)),
            InitCoord(Point(1336, 671), Point(1471, 981)),
            InitCoord(Point(1328, 689), Point(1409, 998)),
            InitCoord(Point(1348, 736), Point(1482, 980)),
            InitCoord(Point(1334, 641), Point(1458, 932)),
            InitCoord(Point(1322, 667), Point(1411, 959)),
            InitCoord(Point(1370, 654), Point(1466, 882)),
            InitCoord(Point(1365, 733), Point(1456, 1050)),
            InitCoord(Point(1312, 519), Point(1389, 864)),
            InitCoord(Point(1250, 639), Point(1373, 953)),
            InitCoord(Point(1345, 769), Point(1441, 984)),
            InitCoord(Point(1322, 546), Point(1430, 841)),
            InitCoord(Point(1284, 723), Point(1369, 1005)),
        ],
        Incisors.UPPER_INNER_LEFT: [
            InitCoord(Point(1383, 732), Point(1526, 1011)),
            InitCoord(Point(1386, 680), Point(1501, 998)),
            InitCoord(Point(1433, 668), Point(1552, 995)),
            InitCoord(Point(1387, 678), Point(1503, 1025)),
            InitCoord(Point(1446, 721), Point(1583, 991)),
            InitCoord(Point(1429, 626), Point(1543, 938)),
            InitCoord(Point(1396, 682), Point(1507, 986)),
            InitCoord(Point(1447, 651), Point(1540, 892)),
            InitCoord(Point(1437, 737), Point(1544, 1055)),
            InitCoord(Point(1386, 557), Point(1488, 865)),
            InitCoord(Point(1347, 681), Point(1459, 960)),
            InitCoord(Point(1429, 775), Point(1514, 996)),
            InitCoord(Point(1405, 542), Point(1505, 865)),
            InitCoord(Point(1370, 697), Point(1465, 1045)),
        ],
        Incisors.UPPER_INNER_RIGHT: [
            InitCoord(Point(1504, 767), Point(1644, 1004)),
            InitCoord(Point(1503, 657), Point(1625, 1006)),
            InitCoord(Point(1544, 652), Point(1678, 986)),
            InitCoord(Point(1515, 669), Point(1614, 1016)),
            InitCoord(Point(1558, 699), Point(1679, 970)),
            InitCoord(Point(1528, 616), Point(1651, 948)),
            InitCoord(Point(1499, 673), Point(1614, 981)),
            InitCoord(Point(1520, 669), Point(1620, 888)),
            InitCoord(Point(1549, 754), Point(1643, 1037)),
            InitCoord(Point(1488, 510), Point(1589, 866)),
            InitCoord(Point(1456, 687), Point(1578, 971)),
            InitCoord(Point(1515, 768), Point(1608, 987)),
            InitCoord(Point(1509, 541), Point(1610, 867)),
            InitCoord(Point(1465, 692), Point(1587, 1022)),
        ],
        Incisors.UPPER_OUTER_RIGHT: [
            InitCoord(Point(1627, 769), Point(1723, 982)),
            InitCoord(Point(1624, 749), Point(1702, 960)),
            InitCoord(Point(1641, 681), Point(1759, 947)),
            InitCoord(Point(1611, 676), Point(1686, 999)),
            InitCoord(Point(1632, 731), Point(1746, 950)),
            InitCoord(Point(1602, 616), Point(1739, 925)),
            InitCoord(Point(1591, 723), Point(1708, 953)),
            InitCoord(Point(1613, 690), Point(1706, 881)),
            InitCoord(Point(1582, 772), Point(1732, 1004)),
            InitCoord(Point(1576, 545), Point(1660, 848)),
            InitCoord(Point(1565, 686), Point(1668, 958)),
            InitCoord(Point(1597, 745), Point(1684, 975)),
            InitCoord(Point(1559, 560), Point(1702, 833)),
            InitCoord(Point(1570, 696), Point(1676, 982)),
        ],
        Incisors.LOWER_OUTER_LEFT: [
            InitCoord(Point(1352, 1026), Point(1445, 1294)),
            InitCoord(Point(1342, 1004), Point(1438, 1299)),
            InitCoord(Point(1352, 1047), Point(1461, 1313)),
            InitCoord(Point(1379, 1059), Point(1452, 1316)),
            InitCoord(Point(1364, 1001), Point(1454, 1284)),
            InitCoord(Point(1382, 982), Point(1483, 1248)),
            InitCoord(Point(1380, 1025), Point(1450, 1310)),
            InitCoord(Point(1355, 893), Point(1429, 1151)),
            InitCoord(Point(1407, 1082), Point(1480, 1356)),
            InitCoord(Point(1347, 919), Point(1426, 1221)),
            InitCoord(Point(1319, 1018), Point(1404, 1294)),
            InitCoord(Point(1400, 985), Point(1470, 1212)),
            InitCoord(Point(1372, 951), Point(1451, 1234)),
            InitCoord(Point(1342, 1071), Point(1429, 1371)),
        ],
        Incisors.LOWER_INNER_LEFT: [
            InitCoord(Point(1440, 1019), Point(1525, 1270)),
            InitCoord(Point(1425, 1013), Point(1503, 1245)),
            InitCoord(Point(1433, 1043), Point(1508, 1293)),
            InitCoord(Point(1446, 1068), Point(1512, 1311)),
            InitCoord(Point(1431, 992), Point(1544, 1274)),
            InitCoord(Point(1462, 973), Point(1540, 1247)),
            InitCoord(Point(1438, 1033), Point(1511, 1291)),
            InitCoord(Point(1418, 903), Point(1489, 1157)),
            InitCoord(Point(1472, 1091), Point(1522, 1347)),
            InitCoord(Point(1418, 918), Point(1488, 1217)),
            InitCoord(Point(1396, 1019), Point(1485, 1278)),
            InitCoord(Point(1463, 1002), Point(1529, 1225)),
            InitCoord(Point(1420, 923), Point(1519, 1217)),
            InitCoord(Point(1421, 1045), Point(1497, 1361)),
        ],
        Incisors.LOWER_INNER_RIGHT: [
            InitCoord(Point(1519, 1009), Point(1601, 1280)),
            InitCoord(Point(1494, 1022), Point(1581, 1231)),
            InitCoord(Point(1495, 1045), Point(1565, 1304)),
            InitCoord(Point(1516, 1057), Point(1583, 1326)),
            InitCoord(Point(1507, 996), Point(1587, 1271)),
            InitCoord(Point(1528, 967), Point(1610, 1239)),
            InitCoord(Point(1487, 1037), Point(1575, 1300)),
            InitCoord(Point(1484, 923), Point(1546, 1161)),
            InitCoord(Point(1507, 1076), Point(1583, 1291)),
            InitCoord(Point(1477, 926), Point(1550, 1214)),
            InitCoord(Point(1478, 1017), Point(1563, 1281)),
            InitCoord(Point(1522, 1001), Point(1601, 1214)),
            InitCoord(Point(1497, 922), Point(1563, 1216)),
            InitCoord(Point(1492, 1044), Point(1569, 1343)),
        ],
        Incisors.LOWER_OUTER_RIGHT: [
            InitCoord(Point(1602, 1012), Point(1693, 1255)),
            InitCoord(Point(1567, 1016), Point(1660, 1279)),
            InitCoord(Point(1546, 1039), Point(1642, 1297)),
            InitCoord(Point(1573, 1049), Point(1651, 1327)),
            InitCoord(Point(1575, 989), Point(1663, 1282)),
            InitCoord(Point(1587, 980), Point(1690, 1245)),
            InitCoord(Point(1531, 1025), Point(1642, 1309)),
            InitCoord(Point(1535, 908), Point(1609, 1173)),
            InitCoord(Point(1575, 1064), Point(1636, 1318)),
            InitCoord(Point(1541, 921), Point(1617, 1215)),
            InitCoord(Point(1543, 1013), Point(1646, 1286)),
            InitCoord(Point(1578, 981), Point(1681, 1208)),
            InitCoord(Point(1555, 932), Point(1630, 1199)),
            InitCoord(Point(1564, 1022), Point(1647, 1316)),
        ],
    }

    @classmethod
    def initialize(cls, asm_dict: AsmDict, images: List[np.ndarray]) -> ImageShapesDict:
        """Initializes a dictionary with imageshapes for each incisor"""
        images = images[: cls._min_hardcoded_length()]
        image_shapes = {
            incisor: cls.find_image_shapes(incisor, asm_dict, images)
            for incisor in Incisors
        }
        return image_shapes

    @classmethod
    def find_image_shapes(
        cls, incisor: Incisors, asm_dict: AsmDict, images: List[np.ndarray]
    ) -> List[ImageShape]:
        """Creates a list of imageshapes for a given incisor"""
        init_coords = cls.IMAGE_COORDS[incisor]
        asm: ActiveShapeModel = asm_dict[incisor]
        return [
            ImageShape(image, asm.mean_shape.conform_to_rect(*init_coords[i]))
            for i, image in enumerate(images)
        ]

    @classmethod
    def _min_hardcoded_length(cls) -> int:
        return min(len(coords) for _k, coords in cls.IMAGE_COORDS.items())
