'''
Copyright (c) 2021-, California Institute of Technology ("Caltech"). U.S.
Government sponsorship acknowledged.
All rights reserved.

Author(s): Alexander Corben
'''


class RasterException(Exception):
    """ Exception within raster processor. """
    pass


class RasterUsageException(Exception):
    """ Exception with usage of raster processor. """
    pass
