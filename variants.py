from model import Model


class NoBet(Model):
    SD_BET = False


class NoCall(Model):
    SD_CALL = False
    SD_PF = False


class NoSD(Model):
    SD_BET = False
    SD_CALL = False
    SD_PF = False


class Loosen(Model):
    SD_CALL = False
    SD_PF = False


class LowR(Model):
    R_OOP = (0.78, 0.85, 0.92)


class V2(Model):
    """The second submission: higher out-of-position realisation."""
    R_OOP = (0.84, 0.89, 0.94)


class LowR2(Model):
    R_OOP = (0.74, 0.83, 0.91)
