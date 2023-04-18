# VASP_ldos_map

Using the DOSCAR output by VASP, the local density of states (LDOS) is interpolated at a distance from the surface of the system via the site projected density of states (DOS). The LDOS is weighted by the tunneling probability to give maps similar to those obtained in scanning tunneling microscopy (STM) experiments. The interpolation of the LDOS is fully parallelizable, with each grid point (or row, for LDOS lines) calculated independently.

The LDOS images are calculated using the method described in: https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03817.
