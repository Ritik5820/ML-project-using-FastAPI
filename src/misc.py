from enum import Enum


class GenderEnum(str, Enum):
    male = "male"
    female = "female"


class RaceEthnicity(str, Enum):
    group_a = "group A"
    group_b = "group B"
    group_c = "group C"
    group_d = "group D"
    group_e = "group E"


class Parental_Level_Of_Eductaion(str, Enum):
    associate_degree = "associate's degree"
    bachelor_degree = "bachelor's degree"
    high_school = "high school"
    master_degree = "master's degree"
    some_college = "some college"
    some_high_school = "some high school"


class Lunch(str, Enum):
    free_or_reduced = "free/reduced"
    standard = "standard"


class TestPreparationCourse(str, Enum):
    none = "none"
    completed = "completed"
