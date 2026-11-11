#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 Aryan Thakkar
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
#

import numpy as np
from gnuradio import gr, gr_unittest, filter


class qa_mmse_fir_interpolator(gr_unittest.TestCase):

    def setUp(self):
        self.tb = gr.top_block()

    def tearDown(self):
        self.tb = None

    def test_001_mmse_fir_interpolator_ff(self):
        intr = filter.mmse_fir_interpolator_ff()
        ntaps = intr.ntaps()
        nsteps = intr.nsteps()
        self.assertEqual(ntaps, 8)
        self.assertEqual(nsteps, 128)

        samples = np.array([0.1, 0.5, 0.8, 1.0, 0.8, 0.5, 0.1, -0.2], dtype=np.float32)

        val_0 = intr.interpolate(samples, 0.0)
        self.assertAlmostEqual(val_0, samples[3], places=5)

        val_half = intr.interpolate(samples, 0.5)
        self.assertTrue(np.isfinite(val_half))

        val_list = intr.interpolate(list(samples), 0.5)
        self.assertAlmostEqual(val_half, val_list, places=5)

        longer_samples = np.array([0.1, 0.5, 0.8, 1.0, 0.8, 0.5, 0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        val_longer = intr.interpolate(longer_samples, 0.5)
        self.assertAlmostEqual(val_half, val_longer, places=5)

        with self.assertRaises(ValueError):
            intr.interpolate([1.0, 2.0, 3.0], 0.5)

        with self.assertRaises(ValueError):
            intr.interpolate(np.ones((2, ntaps), dtype=np.float32), 0.5)

        with self.assertRaises(RuntimeError):
            intr.interpolate(samples, -0.1)
        with self.assertRaises(RuntimeError):
            intr.interpolate(samples, 1.1)

    def test_002_mmse_fir_interpolator_cc(self):
        intr = filter.mmse_fir_interpolator_cc()
        ntaps = intr.ntaps()
        nsteps = intr.nsteps()
        self.assertEqual(ntaps, 8)
        self.assertEqual(nsteps, 128)

        samples = np.array([
            0.1 + 0.2j, 0.5 + 0.1j, 0.8 - 0.3j, 1.0 + 0.5j,
            0.8 + 0.2j, 0.5 - 0.1j, 0.1 + 0.0j, -0.2 + 0.1j
        ], dtype=np.complex64)

        val_0 = intr.interpolate(samples, 0.0)
        self.assertAlmostEqual(val_0.real, samples[3].real, places=5)
        self.assertAlmostEqual(val_0.imag, samples[3].imag, places=5)

        val_half = intr.interpolate(samples, 0.5)
        self.assertTrue(np.isfinite(val_half.real))
        self.assertTrue(np.isfinite(val_half.imag))

        val_list = intr.interpolate(list(samples), 0.5)
        self.assertAlmostEqual(val_half.real, val_list.real, places=5)
        self.assertAlmostEqual(val_half.imag, val_list.imag, places=5)

        with self.assertRaises(ValueError):
            intr.interpolate([1.0 + 0j, 2.0 + 0j], 0.5)

    def test_003_mmse_interp_differentiator_ff(self):
        diff = filter.mmse_interp_differentiator_ff()
        ntaps = diff.ntaps()
        nsteps = diff.nsteps()
        self.assertEqual(ntaps, 8)
        self.assertEqual(nsteps, 128)

        samples = np.array([0.1, 0.5, 0.8, 1.0, 0.8, 0.5, 0.1, -0.2], dtype=np.float32)

        val_half = diff.differentiate(samples, 0.5)
        self.assertTrue(np.isfinite(val_half))

        val_list = diff.differentiate(list(samples), 0.5)
        self.assertAlmostEqual(val_half, val_list, places=5)

        with self.assertRaises(ValueError):
            diff.differentiate([1.0, 2.0, 3.0], 0.5)

    def test_004_mmse_interp_differentiator_cc(self):
        diff = filter.mmse_interp_differentiator_cc()
        ntaps = diff.ntaps()
        nsteps = diff.nsteps()
        self.assertEqual(ntaps, 8)
        self.assertEqual(nsteps, 128)

        samples = np.array([
            0.1 + 0.2j, 0.5 + 0.1j, 0.8 - 0.3j, 1.0 + 0.5j,
            0.8 + 0.2j, 0.5 - 0.1j, 0.1 + 0.0j, -0.2 + 0.1j
        ], dtype=np.complex64)

        val_half = diff.differentiate(samples, 0.5)
        self.assertTrue(np.isfinite(val_half.real))
        self.assertTrue(np.isfinite(val_half.imag))

        val_list = diff.differentiate(list(samples), 0.5)
        self.assertAlmostEqual(val_half.real, val_list.real, places=5)
        self.assertAlmostEqual(val_half.imag, val_list.imag, places=5)

        with self.assertRaises(ValueError):
            diff.differentiate([1.0 + 0j, 2.0 + 0j], 0.5)


if __name__ == '__main__':
    gr_unittest.run(qa_mmse_fir_interpolator)
