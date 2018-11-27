# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

"""
This module defines a workflow for adsorption on surfaces
"""

import numpy as np

from fireworks import Workflow

from atomate.vasp.fireworks.core import OptimizeFW, TransmuterFW

from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core.surface import generate_all_slabs, Slab
from pymatgen.transformations.advanced_transformations import SlabTransformation
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.io.vasp.sets import MVLSlabSet
from pymatgen import Structure

__author__ = 'Joseph Montoya, Richard Tran'
__email__ = 'montoyjh@lbl.gov'


# TODO: Add framework for including vibrations and free energy
def get_slab_fw(slab, transmuter=False, db_file=None, vasp_input_set=None,
                parents=None, vasp_cmd="vasp", name="", additional_fields=None):
    """
    Function to generate a a slab firework.  Returns a TransmuterFW if
    bulk_structure is specified, constructing the necessary transformations
    from the slab and slab generator parameters, or an OptimizeFW if only a
    slab is specified.

    Args:
        slab (Slab or Structure): structure or slab corresponding
            to the slab to be calculated
        transmuter (bool): whether or not to use a TransmuterFW based
            on slab params, if this option is selected, input slab must
            be a Slab object (as opposed to Structure)
        vasp_input_set (VaspInputSet): vasp_input_set corresponding to
            the slab calculation
        parents (Fireworks or list of ints): parent FWs
        db_file (string): path to database file
        vasp_cmd (string): vasp command
        additional_fields (dict): dict of additional fields to add

    Returns:
        Firework corresponding to slab calculation
    """
    vasp_input_set = vasp_input_set or MVLSlabSet(slab)

    # If a bulk_structure is specified, generate the set of transformations,
    # else just create an optimize FW with the slab
    if transmuter:
        if not isinstance(slab, Slab):
            raise ValueError("transmuter mode requires slab to be a Slab object")

        # Get transformation from oriented bulk and slab
        oriented_bulk = slab.oriented_unit_cell
        slab_trans_params = get_slab_trans_params(slab)
        trans_struct = SlabTransformation(**slab_trans_params)
        slab_from_bulk = trans_struct.apply_transformation(oriented_bulk)

        # Ensures supercell construction
        supercell_trans = SupercellTransformation.from_scaling_factors(
            round(slab.lattice.a / slab_from_bulk.lattice.a),
            round(slab.lattice.b / slab_from_bulk.lattice.b))

        # Get site properties, set velocities to zero if not set to avoid
        # custodian issue
        site_props = slab.site_properties
        if 'velocities' not in site_props:
            site_props['velocities'] = [0. for s in slab]

        # Get adsorbates for InsertSitesTransformation
        if "adsorbate" in slab.site_properties.get("surface_properties", ""):
            ads_sites = [site for site in slab
                         if site.properties["surface_properties"] == "adsorbate"]
        else:
            ads_sites = []
        transformations = [
            "SlabTransformation", "SupercellTransformation",
            "InsertSitesTransformation", "AddSitePropertyTransformation"]
        trans_params = [slab_trans_params,
                        {"scaling_matrix": supercell_trans.scaling_matrix},
                        {"species": [site.species_string for site in ads_sites],
                         "coords": [site.frac_coords for site in ads_sites]},
                        {"site_properties": site_props}]
        fw = TransmuterFW(name=name, structure=oriented_bulk,
                          transformations=transformations,
                          transformation_params=trans_params,
                          copy_vasp_outputs=True, db_file=db_file,
                          vasp_cmd=vasp_cmd, parents=parents,
                          vasp_input_set=vasp_input_set)
    else:
        fw = OptimizeFW(name=name, structure=slab,
                        vasp_input_set=vasp_input_set, vasp_cmd=vasp_cmd,
                        db_file=db_file, parents=parents, job_type="normal")
    # Add slab metadata
    fw.tasks[-1]["additional_fields"].update({"slab": slab})
    if additional_fields:
        fw.tasks[-1]["additional_fields"].update(additional_fields)

    return fw


def get_slab_trans_params(slab):
    """
    Gets a set of slab transformation params

    Args:
        slab (Slab): slab to find transformation params from

    Returns (SlabTransformation):
        Transformation for a transformation that will transform
        the oriented unit cell to the slab
    """
    slab = slab.copy()
    if slab.site_properties.get("surface_properties"):
        adsorbate_indices = [slab.index(s) for s in slab if
                             s.properties['surface_properties'] == 'adsorbate']
        slab.remove_sites(adsorbate_indices)

    # Note: this could fail if the slab is non-contiguous in the c direction,
    # i. e. if sites are translated through the pbcs
    heights = [np.dot(s.coords, slab.normal) for s in slab]
    slab_thickness = np.abs(max(heights) - min(heights))
    bulk_a, bulk_b, bulk_c = slab.oriented_unit_cell.lattice.matrix
    bulk_normal = np.cross(bulk_a, bulk_b)
    bulk_normal /= np.linalg.norm(bulk_normal)
    bulk_height = np.abs(np.dot(bulk_normal, bulk_c))
    slab_cell_height = np.abs(np.dot(slab.lattice.matrix[2], slab.normal))

    total_layers = slab_cell_height / bulk_height
    slab_layers = np.ceil(slab_thickness / slab_cell_height * total_layers)
    vac_layers = total_layers - slab_layers

    min_slab_size = slab_cell_height * slab_layers / total_layers - 0.03
    min_vac_size = slab_cell_height * vac_layers / total_layers - 0.03
    # params = {"miller_index": [0, 0, 1], "shift": slab.shift,
    #           "min_slab_size": min_slab_size, "min_vacuum_size": min_vac_size}
    # trans = SlabTransformation(**params)
    # new_slab = trans.apply_transformation(slab.oriented_unit_cell)
    # if slab.composition.reduced_formula == "Si":
    #     import nose; nose.tools.set_trace()

    return {"miller_index": slab.miller_index, "shift": slab.shift,
            "min_slab_size": min_slab_size, "min_vacuum_size": min_vac_size}


def get_wf_slab(slab, include_bulk_opt=False, adsorbates=None,
                ads_structures_params=None, vasp_cmd="vasp",
                db_file=None, add_molecules_in_box=False, additional_fields=None):
    """
    Gets a workflow corresponding to a slab calculation along with optional
    adsorbate calcs and precursor oriented unit cell optimization

    Args:
        slabs (list of Slabs or Structures): slabs to calculate
        include_bulk_opt (bool): whether to include bulk optimization,
            this flag sets the slab fireworks to be TransmuterFWs based
            on bulk optimization of oriented unit cells
        adsorbates ([Molecule]): list of molecules to place as adsorbates
        ads_structures_params (dict): parameters to be supplied as
            kwargs to AdsorbateSiteFinder.generate_adsorption_structures
        add_molecules_in_box (boolean): flag to add calculation of
            adsorbate molecule energies to the workflow
        db_file (string): path to database file
        vasp_cmd (string): vasp command
        additional_fields (dict): dict of additional fields to add

    Returns:
        Workflow
    """
    fws, parents = [], []

    if additional_fields is None:
        additional_fields = {}

    if adsorbates is None:
        adsorbates = []

    if ads_structures_params is None:
        ads_structures_params = {}

    # Add bulk opt firework if specified
    if include_bulk_opt:
        oriented_bulk = slab.oriented_unit_cell
        vis = MVLSlabSet(oriented_bulk, bulk=True)
        name = slab.composition.reduced_formula
        if getattr(slab, "miller_index", None):
            name += "_{}".format(slab.miller_index)
        fws.append(OptimizeFW(name="{} ouc optimization".format(name),
                              structure=oriented_bulk, vasp_input_set=vis,
                              vasp_cmd=vasp_cmd, db_file=db_file))
        fws[-1].tasks[-1]["additional_fields"].update(additional_fields)
        parents = fws[-1]

    name = slab.composition.reduced_formula
    if getattr(slab, "miller_index", None):
        name += "_{}".format(slab.miller_index)
    # Create slab fw and add it to list of fws
    slab_fw = get_slab_fw(slab, include_bulk_opt, db_file=db_file,
                          vasp_cmd=vasp_cmd, parents=parents,
                          name="{} slab optimization".format(name),
                          additional_fields=additional_fields)
    fws.append(slab_fw)

    for adsorbate in adsorbates:
        ads_slabs = AdsorbateSiteFinder(slab).generate_adsorption_structures(
            adsorbate, **ads_structures_params)
        for n, ads_slab in enumerate(ads_slabs):
            # Create adsorbate fw
            ads_name = "{}-{} adsorbate optimization {}".format(
                adsorbate.composition.formula, name, n)
            adsorbate_fw = get_slab_fw(
                ads_slab, include_bulk_opt, db_file=db_file, vasp_cmd=vasp_cmd,
                parents=parents, name=ads_name, additional_fields=additional_fields)
            fws.append(adsorbate_fw)

    if isinstance(slab, Slab):
        name = "{}_{} slab workflow".format(
            slab.composition.reduced_composition, slab.miller_index)
    else:
        name = "{} slab workflow".format(slab.composition.reduced_composition)

    wf = Workflow(fws, name=name)

    # Add optional molecules workflow
    if add_molecules_in_box:
        molecule_wf = get_wf_molecules(adsorbates, db_file=db_file,
                                       vasp_cmd=vasp_cmd)
        wf.append_wf(molecule_wf)

    return wf


def get_wf_molecules(molecules, vasp_input_set=None, db_file=None,
                     vasp_cmd="vasp", name=""):
    """
    Args:
        molecules (Molecules): list of molecules to calculate
        vasp_input_set (DictSet): VaspInputSet for molecules
        db_file (string): database file path
        vasp_cmd (string): VASP command
        name (string): name for workflow

    Returns:
        workflow consisting of molecule calculations
    """
    fws = []

    for molecule in molecules:
        # molecule in box
        m_struct = molecule.get_boxed_structure(10, 10, 10,
                                                offset=np.array([5, 5, 5]))
        vis = vasp_input_set or MVLSlabSet(m_struct)
        fws.append(OptimizeFW(structure=molecule, job_type="normal",
                              vasp_input_set=vis, db_file=db_file,
                              vasp_cmd=vasp_cmd))
    name = name or "molecules workflow"
    return Workflow(fws, name=name)


# TODO: this will duplicate a precursor optimization for slabs with
#       the same miller index, but different shift
def get_wfs_all_slabs(bulk_structure, include_bulk_opt=False,
                      adsorbates=None, max_index=1, slab_gen_params=None,
                      ads_structures_params=None, vasp_cmd="vasp",
                      db_file=None, add_molecules_in_box=False,
                      inc_reconstructions=False, additional_fields=None):
    """
    Convenience constructor that allows a user to construct a workflow
    that finds all adsorption configurations (or slabs) for a given
    max miller index.

    Args:
        bulk_structure (Structure): bulk structure from which to construct slabs
        include_bulk_opt (bool): whether to include bulk optimization
            of oriented unit cells
        adsorbates ([Molecule]): adsorbates to place on surfaces
        max_index (int): max miller index
        slab_gen_params (dict): dictionary of kwargs for generate_all_slabs
        ads_structures_params (dict): dictionary of kwargs for generating
            of adsorption structures via AdsorptionSiteFinder
        vasp_cmd (str): vasp command
        db_file (str): location of db file
        add_molecules_in_box (bool): whether to add molecules in a box
            for the entire workflow
        inc_reconstructions (bool): Whether to include reconstructed slabs in
            the workflow. generate_all_slabs will look for all possible reconstructions
            available in a json file available on pymatgen/core/reconstruction_archives.json
        additional_fields (dict): dict of additional fields to add

    Returns:
        list of slab-specific Workflows
    """
    # We define three default sgp (parameters for SlabGenerator)
    reduced_bulk = bulk_structure.get_primitive_structure(constrain_latt=[False, False, True,
                                                                          False, False, False])
    if slab_gen_params:
        # Use slab_gen_params if user provided
        sgp = slab_gen_params
    elif (len(reduced_bulk)/bulk_structure.lattice.c)*10 >= 8:
        # Use a minimum of 10 Å vacuum and slab for systems that
        # have at least 8 atoms under a 10 Å slab along (001)
        sgp = {"min_slab_size": 10, "min_vacuum_size": 10,
               "max_normal_search": 1,
               "center_slab": True}
    else:
        # Use a minimum of 4 dhkl layers in the vacuum and slab for systems
        # that have less than 8 atoms under a 10 Å slab along (001)
        sgp = {"min_slab_size": 4, "min_vacuum_size": 4,
               "in_unit_planes": True, "max_normal_search": 1,
               "center_slab": True}

    if inc_reconstructions:
        sgp['include_reconstructions'] = True
    slabs = generate_all_slabs(bulk_structure, max_index=max_index, **sgp)
    wfs = []
    for slab in slabs:
        slab_wf = get_wf_slab(slab, include_bulk_opt, adsorbates,
                              ads_structures_params, vasp_cmd, db_file,
                              additional_fields=additional_fields)
        wfs.append(slab_wf)

    if add_molecules_in_box:
        wfs.append(get_wf_molecules(adsorbates, db_file=db_file,
                                    vasp_cmd=vasp_cmd))
    return wfs

