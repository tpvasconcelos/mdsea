from scipy.constants import codata

# TODO: would be nice to make this a frozen-dict
physical_constants = {
    "k_boltzmann": codata.value("Boltzmann constant"),
    "gravity_acceleration": codata.value("standard acceleration of gravity"),
}
