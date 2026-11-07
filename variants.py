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
