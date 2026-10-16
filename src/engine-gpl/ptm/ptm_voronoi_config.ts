/*
 * Ported from LAMMPS (https://github.com/lammps/lammps, commit b332f28): the voro++-derived Voronoi cell of the PTM library (src/PTM/ptm_voronoi_cell.cpp, ptm_voronoi_cell.h, ptm_voronoi_config.h), modified by PM Larsen for Polyhedral Template Matching. voro++ is Copyright (c) 2008, The Regents of the University of California (BSD-3, see LICENSE.voro++ next to this file). The LAMMPS compute that uses it (src/PTM/compute_ptm_atom.cpp) is GPL-2.0; this file is GPL-2.0 too (src/engine-gpl/LICENSE).
 */
/*
Voro++ Copyright (c) 2008, The Regents of the University of California, through
Lawrence Berkeley National Laboratory (subject to receipt of any required
approvals from the U.S. Dept. of Energy). All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

(1) Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

(3) Neither the name of the University of California, Lawrence Berkeley
National Laboratory, U.S. Dept. of Energy nor the names of its contributors may
be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
("Enhancements") to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to Lawrence Berkeley National
Laboratory, without imposing a separate written license agreement for such
Enhancements, then you hereby grant the following license: a  non-exclusive,
royalty-free perpetual license to install, use, modify, prepare derivative
works, incorporate into other computer software, distribute, and sublicense
such enhancements or derivative works thereof, in binary and source code form.
*/


// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011
//
// Modified by PM Larsen for use in Polyhedral Template Matching

// Master configuration file for setting various compile-time options (config.hh).
// The C preprocessor macros are exported as constants; the C namespace is ptm_voro.

// These constants set the initial memory allocation for the Voronoi cell
/** The initial memory allocation for the number of vertices. */
export const init_vertices = 256;
/** The initial memory allocation for the maximum vertex order. */
export const init_vertex_order = 64;
/** The initial memory allocation for the number of regular vertices of order 3. */
export const init_3_vertices = 256;
/** The initial memory allocation for the number of vertices of higher order. */
export const init_n_vertices = 8;
/** The initial buffer size for marginal cases used by the suretest class. */
export const init_marginal = 64;
/** The initial size for the delete stack. */
export const init_delete_size = 256;
/** The initial size for the auxiliary delete stack. */
export const init_delete2_size = 256;
/** The initial size for the wall pointer array. */
export const init_wall_size = 32;
/** The default initial size for the ordering class. */
export const init_ordering_size = 4096;
/** The initial size of the pre_container chunk index. */
export const init_chunk_size = 256;

// If the initial memory is too small, the program dynamically allocates more.
// However, if the limits below are reached, then the program bails out.
/** The maximum memory allocation for the number of vertices. */
export const max_vertices = 16777216;
/** The maximum memory allocation for the maximum vertex order. */
export const max_vertex_order = 2048;
/** The maximum memory allocation for the any particular order of vertex. */
export const max_n_vertices = 16777216;
/** The maximum buffer size for marginal cases used by the suretest class. */
export const max_marginal = 16777216;
/** The maximum size for the delete stack. */
export const max_delete_size = 16777216;
/** The maximum size for the auxiliary delete stack. */
export const max_delete2_size = 16777216;
/** The maximum amount of particle memory allocated for a single region. */
export const max_particle_memory = 16777216;
/** The maximum size for the wall pointer array. */
export const max_wall_size = 2048;
/** The maximum size for the ordering class. */
export const max_ordering_size = 67108864;
/** The maximum size for the pre_container chunk index. */
export const max_chunk_size = 65536;

/** The chunk size in the pre_container classes. */
export const pre_container_chunk_size = 1024;

/**
 * Voro++ status/debug verbosity (#define VOROPP_VERBOSE 0 in C). At level 0 no
 * messages are printed, so the verbose-only branches of the C code are not
 * translated.
 */
export const VOROPP_VERBOSE = 0;

/** If a point is within this distance of a cutting plane, then the code
 * assumes that point exactly lies on the plane. */
export const tolerance = 1e-11;

/** If a point is within this distance of a cutting plane, then the code stores
 * whether this point is inside, outside, or exactly on the cutting plane in
 * the marginal cases buffer. */
export const tolerance2 = 2e-11;

/** The square of the tolerance. */
export const tolerance_sq = tolerance * tolerance;

/** A large number that is used in the computation. */
export const large_number = 1e30;

/** A radius to use as a placeholder when no other information is available. */
export const default_radius = 0.5;

/** The maximum number of shells of periodic images to test over. */
export const max_unit_voro_shells = 10;

/** A guess for the optimal number of particles per block. */
export const optimal_particles = 5.6;

/** If this is set to 1, then the code reports any instances of particles being
 * put outside of the container geometry. */
export const VOROPP_REPORT_OUT_OF_BOUNDS = 0;

/** Voro++ returns this status code if there is a file-related error. */
export const VOROPP_FILE_ERROR = 1;

/** Voro++ returns this status code if there is a memory allocation error. */
export const VOROPP_MEMORY_ERROR = 2;

/** Voro++ returns this status code if there is any type of internal error. */
export const VOROPP_INTERNAL_ERROR = 3;

/** Voro++ returns this status code if it could not interpret the command-line arguments. */
export const VOROPP_CMD_LINE_ERROR = 4;
