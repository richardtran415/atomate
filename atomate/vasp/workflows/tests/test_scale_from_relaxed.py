# coding: utf-8

from __future__ import division, print_function, unicode_literals, absolute_import

import os
import unittest

from fireworks import FWorker
from fireworks.core.rocket_launcher import rapidfire
from pymatgen.core import Lattice
from atomate.vasp.powerups import use_fake_vasp
from atomate.utils.testing import AtomateTest
from atomate.vasp.workflows.base.scale_from_relaxed import get_wf_from_unrelaxed
from pymatgen.core import Structure

__author__ = 'Richard Tran, Hui Zheng'
__email__ = 'rit001@eng.ucsd.edu'

module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(module_dir, "..", "..", "..", "common", "test_files")
ref_dir = os.path.join(module_dir, "..", "..", "test_files")

DEBUG_MODE = False  # If true, retains the database and output dirs at the end of the test
VASP_CMD = None  # If None, runs a "fake" VASP. Otherwise, runs VASP with this command...


class TestRelaxationScalingWorkflow(AtomateTest):
    def setUp(self, lpad=True):
        super(TestRelaxationScalingWorkflow, self).setUp()
        self.initial_base_structure = Structure.from_file("Fe_gb_init.cif")
        similar_structures = [Structure.from_file("Mo_gb_init.cif"),
                              Structure.from_file("Ta_gb_init.cif"), ]

        self.wf = get_wf_from_unrelaxed(self.initial_base_structure,
                                        similar_structures,
                                        db_file=os.path.join(db_dir, "db.json"))

    @classmethod
    def _simulate_vasprun_start_from_unrelaxed(cls, wf):

        bcc_gbs_ref_dir = {"Fe_Structure_initial scaling optimization": "1",
                           "Mo_Structure_from_Fe_Structure scaling optimization": "2",
                           "Ta_Structure_from_Ta_Structure scaling optimization": "3"}

        return use_fake_vasp(wf, bcc_gbs_ref_dir)

    def _check_run(self, d, mode):
        if mode not in ["Fe_Structure_base_relaxation scaling optimization",
                        "Mo_Structure_scaled_from_Fe_Structure scaling optimization",
                        "Ta_Structure_from_Ta_Structure scaling optimization"]:
            raise ValueError("Invalid mode!")
        if "scaling optimization" in mode:
            self.assertTrue(d["formula_reduced_abc"] in ["Fe", "Mo", "Ta"])

    def test_start_from_unrelaxed_wf(self):
        wf = self._simulate_vasprun_start_from_bulk(self.wf)
        self.assertEqual(len(self.wf.fws), 3)
        self.lp.add_wf(wf)
        rapidfire(self.lp, fworker=FWorker(env={"db_file": os.path.join(db_dir, "db.json")}))

        # check relaxation
        d = self.get_task_collection().find_one({"task_label": "Fe_Structure_base_relaxation scaling optimization"})
        self._check_run(d, mode="Fe_Structure_base_relaxation scaling optimization")
        # check relaxation
        d = self.get_task_collection().find_one(
            {"task_label": "Mo_Structure_scaled_from_Fe_Structure scaling optimization"})
        self._check_run(d, mode="Mo_Structure_scaled_from_Fe_Structure scaling optimization")
        # check relaxation
        d = self.get_task_collection().find_one({"task_label": "Ta_Structure_from_Ta_Structure scaling optimization"})
        self._check_run(d, mode="Ta_Structure_from_Ta_Structure scaling optimization")

        wf = self.lp.get_wf_by_fw_id(1)
        self.assertTrue(all([s == 'COMPLETED' for s in wf.fw_states.values()]))


if __name__ == "__main__":
    unittest.main()
