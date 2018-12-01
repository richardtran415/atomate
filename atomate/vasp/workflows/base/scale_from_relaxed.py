# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines a workflow for the optimization of similar structures using
the relaxed configuration of a base structure as the initial configuration.
"""

from fireworks import Workflow

from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW

from pymatgen.transformations.standard_transformations import ScaleToRelaxedTransformation

__author__ = 'Richard Tran, Hui Zheng'
__email__ = 'rit001@eng.ucsd.edu'


def get_scale_from_relaxed_fw(unrelaxed_structure, relaxed_structure, similar_structure,
                              species_map=None, vasp_cmd="vasp", db_file=None,
                              vasp_input_set=None, parents=None):
    """
    Gets firework that relaxes a structure whose initial configuration emulates a similar
    relaxed base structure.

    Args:
        unrelaxed_structure (Structure): Initial, unrelaxed structure
        relaxed_structure (Structure): Relaxed structure
        species_map (dict): A dict or list of tuples containing the species mapping in
            string-string pairs. The first species corresponds to the relaxed
            structure while the second corresponds to the species in the
            structure to be scaled. E.g., {"Li":"Na"} or [("Fe2+","Mn2+")].
            Multiple substitutions can be done. Overloaded to accept
            sp_and_occu dictionary E.g. {"Si: {"Ge":0.75, "C":0.25}},
            which substitutes a single species with multiple species to
            generate a disordered structure.
        similar_structure (Structure): Struture similar (same number of sites and crystal
            type) to the base_structure. This structure will be scaled in a way to emulate
            the final relaxed state of the base_structure prior to optimization.
        vasp_input_set (VaspInputSet): input set to use. Defaults to MPRelaxSet() if None.
        db_file (string): path to database file
        vasp_cmd (string): vasp command
        parents (Fireworks or list of ints): parent FWs

    Returns:
        Firework corresponding to similar_structure optimization
    """
    vasp_input_set = vasp_input_set or MPRelaxSet(similar_structure)

    if parents is None:
        parents = []

    name += "%s_%s_scaled_from_relaxed_%s_%s" % (similar_structure.composition.reduced_formula,
                                                 type(similar_structure).__name__,
                                                 relaxed_structure.composition.reduced_formula,
                                                 type(relaxed_structure).__name__)

    fw = TransmuterFW(similar_structure, name=name,
                      transformations=['ScaleToRelaxedTransformation'],
                      transformation_params={"unrelaxed_structure": unrelaxed_structure,
                                             "relaxed_structure": relaxed_structure,
                                             "species_map": species_map},
                      copy_vasp_outputs=True, db_file=db_file,
                      vasp_cmd=vasp_cmd, parents=parents,
                      vasp_input_set=vasp_input_set)

    return fw


def get_wfs_from_relaxed(unrelaxed_structure, relaxed_structure, similar_structures,
                         species_map=None, vasp_cmd="vasp", db_file=None,
                         vasp_input_set=None, parents=None):
    """
    Gets a workflow from the relaxed base structure. The user provided initial and
    final structure are used to determine the amount of site and lattice relaxation
    to apply to other structures.

    Args:
        unrelaxed_structure (Structure): Initial, unrelaxed structure
        relaxed_structure (Structure): Relaxed structure
        species_map (dict): A dict or list of tuples containing the species mapping in
            string-string pairs. The first species corresponds to the relaxed
            structure while the second corresponds to the species in the
            structure to be scaled. E.g., {"Li":"Na"} or [("Fe2+","Mn2+")].
            Multiple substitutions can be done. Overloaded to accept
            sp_and_occu dictionary E.g. {"Si: {"Ge":0.75, "C":0.25}},
            which substitutes a single species with multiple species to
            generate a disordered structure.
        similar_structures (list of Structure): List of strutures to be scaled in a way
            to emulate the final relaxed state of the base structure prior to optimization.
        vasp_input_set (VaspInputSet): input set to use. Defaults to MPRelaxSet() if None.
        db_file (string): path to database file
        vasp_cmd (string): vasp command
        parents (Fireworks or list of ints): parent FWs

    Returns:
        Workflow
    """
    fws = []

    vasp_input_set = vasp_input_set or MPRelaxSet(similar_structures[0])

    for s in similar_structures:
        vasp_input_set.structure = s
        fw = get_scale_from_relaxed_fw(unrelaxed_structure, relaxed_structure, s,
                                       species_map=species_map, vasp_cmd=vasp_cmd,
                                       db_file=db_file, vasp_input_set=vasp_input_set,
                                       parents=parents)
        fws.append(fw)

    name += "scale_from_relaxed_%s_%s" % (relaxed_structure.composition.reduced_formula,
                                          type(relaxed_structure).__name__)

    wf = Workflow(fws, name=name)

    return wf
