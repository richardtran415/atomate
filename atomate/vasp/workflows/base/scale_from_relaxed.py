# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines a workflow for the optimization of similar structures using
the relaxed configuration of a base structure as the initial configuration.
"""

from fireworks import FiretaskBase, explicit_serialize
from fireworks import Workflow, Firework

from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW
from atomate.common.firetasks.glue_tasks import CopyFilesFromCalcLoc

from pymatgen.transformations.standard_transformations import ScaleToRelaxedTransformation
from pymatgen.io.vasp.sets import MPRelaxSet
from pymatgen import Structure

__author__ = 'Richard Tran, Hui Zheng'
__email__ = 'rit001@eng.ucsd.edu'


@explicit_serialize
class WriteScaledStructureIOSet(FiretaskBase):
    """
    Apply the provided transformations to the input structure and write the
    input set for that structure. Reads structure from POSCAR if no structure provided. Note that
    if a transformation yields many structures from one, only the last structure in the list is
    used.

    Required params:
        structure (Structure): input structure
        transformations (list): list of names of transformation classes as defined in
            the modules in pymatgen.transformations
        vasp_input_set (VaspInputSet): VASP input set.

    Optional params:
        transformation_params (list): list of dicts where each dict specifies the input parameters
            to instantiate the transformation class in the transformations list.
        override_default_vasp_params (dict): additional user input settings.
        prev_calc_dir: path to previous calculation if using structure from another calculation.
    """

    required_params = ["similar_structure", "unrelaxed_structure", "vasp_input_set"]
    optional_params = ["relaxed_structure"]

    def run_task(self, fw_spec):

        transformations = [ScaleToRelaxedTransformation]
        transformation_params = {"unrelaxed_structure": self["unrelaxed_structure"],
                                 "relaxed_structure": relaxed_structure,
                                 "species_map": species_map}

        for t in self["transformations"]:
            found = False
            for m in ["advanced_transformations", "defect_transformations",
                      "site_transformations", "standard_transformations"]:
                mod = import_module("pymatgen.transformations.{}".format(m))
                try:
                    t_cls = getattr(mod, t)
                except AttributeError:
                    continue
                t_obj = t_cls(**transformation_params.pop(0))
                transformations.append(t_obj)
                found = True
            if not found:
                raise ValueError("Could not find transformation: {}".format(t))

        structure = self['similar_structure']
        ts = TransformedStructure(structure)
        transmuter = StandardTransmuter([ts], transformations)
        final_structure = transmuter.transformed_structures[-1].final_structure.copy()
        vis_orig = self["vasp_input_set"]
        vis_dict = vis_orig.as_dict()
        vis_dict["structure"] = final_structure.as_dict()
        vis = vis_orig.__class__.from_dict(vis_dict)
        vis.write_input(".")

        dumpfn(transmuter.transformed_structures[-1], "transformations.json")


def get_scale_from_relaxed_fw(unrelaxed_structure, relaxed_structure, similar_structure,
                              species_map=None, vasp_cmd="vasp", db_file=None,
                              vasp_input_set=None, parents=None):
    """
    Gets firework that relaxes a structure whose initial configuration emulates a similar
    relaxed base structure. If a parent exists, get the relaxed_structure from the parent
    calculation rather than the user input.

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
    name = "%s_scaled_from_relaxed_%s_%s" % (type(similar_structure).__name__,
                                             unrelaxed_structure.composition.reduced_formula,
                                             type(unrelaxed_structure).__name__)

    fw =  TransmuterFW(similar_structure, name=name,
                       transformations=['ScaleToRelaxedTransformation'],
                       transformation_params={"unrelaxed_structure": unrelaxed_structure,
                                              "relaxed_structure": relaxed_structure,
                                              "species_map": species_map},
                       copy_vasp_outputs=False, db_file=db_file,
                       vasp_cmd=vasp_cmd, parents=parents,
                       vasp_input_set=vasp_input_set)

    fw.tasks[0] = WriteScaledStructureIOSet(similar_structure=similar_structure,
                                            unrelaxed_structure=unrelaxed_structure,
                                            vasp_input_set=vasp_input_set,
                                            relaxed_structure=relaxed_structure)
    return fw
    # fw = fw.as_dict()
    # fw['spec']['_tasks'][0] = WriteScaledStructureIOSet(similar_structure=similar_structure,
    #                                                     unrelaxed_structure=unrelaxed_structure,
    #                                                     vasp_input_set=vasp_input_set,
    #                                                     relaxed_structure=relaxed_structure)
    #
    # return Firework.from_dict(fw)


def get_wf_from_relaxed(unrelaxed_structure, relaxed_structure, similar_structures,
                        vasp_cmd="vasp", db_file=None, vasp_input_set=None, parents=None):
    """
    Gets a workflow from the relaxed base structure. The user provided initial and
    final structure are used to determine the amount of site and lattice relaxation
    to apply to other structures.

    Args:
        unrelaxed_structure (Structure): Initial, unrelaxed structure
        relaxed_structure (Structure): Relaxed structure
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
    if parents:
        fws.extend(parents)
    vasp_input_set = vasp_input_set or MPRelaxSet(similar_structures[0])

    for s in similar_structures:
        vasp_input_set.structure = s
        fw = get_scale_from_relaxed_fw(unrelaxed_structure, relaxed_structure, s,
                                       vasp_cmd=vasp_cmd, db_file=db_file,
                                       vasp_input_set=vasp_input_set, parents=parents)
        fws.append(fw)

    name = "scale_from_relaxed_%s_%s" % (unrelaxed_structure.composition.reduced_formula,
                                         type(unrelaxed_structure).__name__)

    wf = Workflow(fws, name=name)

    return wf


def get_wf_from_unrelaxed(base_structure, similar_structures, vasp_cmd="vasp",
                          vasp_input_set=None, db_file=None, parents=None):
    """
    Gets a workflow that relaxes the base structure. Afterwards, a list of
    similar structures will have their sites and lattice scaled in a way to
    emulate the relaxation of the base structure. Though it is not always a
    guarantee, the goal is to get a set of initial structures closer to
    the final relaxed state before running calculations.

    Args:
        base_structure (Structure): This structure will be fully relaxed.
            Its initial and final structure will then be used to determine
            the relaxation behavior of the sites and lattice in similar structures
        similar_structures (list of Structures): List of structures that are
            similar (same number of sites and crystal type) to the base_structure.
            These structures will be scaled in a way to emulate the final relaxed
            state of the base_structure prior to optimization.
        vasp_input_set (VaspInputSet): input set to use. Defaults to MPRelaxSet() if None.
        db_file (string): path to database file
        vasp_cmd (string): vasp command
        parents (Fireworks or list of ints): parent FWs

    Returns:
        Workflow
    """
    vasp_input_set = vasp_input_set or MPRelaxSet(base_structure)
    name = "%s_base_relaxation" % (type(base_structure).__name__)
    optFW = OptimizeFW(base_structure, vasp_input_set=vasp_input_set, name=name,
                       vasp_cmd=vasp_cmd, db_file=db_file, parents=parents)
    optFW = optFW.as_dict()
    t = CopyFilesFromCalcLoc(calc_loc=name, filenames=["CONTCAR"]).as_dict()
    optFW['spec']['_tasks'].append(t)
    optFW = OptimizeFW.from_dict(optFW)
    return get_wf_from_relaxed(base_structure, None, similar_structures,
                                vasp_cmd=vasp_cmd, db_file=db_file, parents=[optFW],
                                vasp_input_set=vasp_input_set)