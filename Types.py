from enum import Enum, auto

class AutoName(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name
    
    @staticmethod
    def listAllTypes(enum: Enum):
        names = [enumType.name for enumType in enum]
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return " or ".join(names)
        else:
            names[-1] = "or " + names[-1]
            return ", ".join(names)

class Flavor(AutoName):
    TFT = auto(),
    EDLT = auto(),
    OECT = auto()

class Ttype(AutoName):
    n = auto(),
    p = auto()

class DeformMode(AutoName):
    uniaxial_L= auto(),
    uniaxial_W = auto(),
    biaxial_WL = auto()