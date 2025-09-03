/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

        m.def("rasterize_gaussians", &RasterizeGaussiansCUDA, "Rasterize Gaussians",
                py::arg("background"),
                py::arg("means3D"),
                py::arg("colors"),
                py::arg("opacity"),
                py::arg("scales"),
                py::arg("rotations"),
                py::arg("scale_modifier"),
                py::arg("cov3D_precomp"),
                py::arg("viewmatrix"),
                py::arg("projmatrix"),
                py::arg("tan_fovx"),
                py::arg("tan_fovy"),
                py::arg("image_height"),
                py::arg("image_width"),
                py::arg("sh"),
                py::arg("degree"),
                py::arg("campos"),
                py::arg("prefiltered"),
                py::arg("antialiasing"),
                py::arg("debug"),
                py::arg("axis")
        );

        m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA, "Rasterize Gaussians backward",
                py::arg("background"),
                py::arg("means3D"),
                py::arg("radii"),
                py::arg("colors"),
                py::arg("opacity"),
                py::arg("scales"),
                py::arg("rotations"),
                py::arg("scale_modifier"),
                py::arg("cov3D_precomp"),
                py::arg("viewmatrix"),
                py::arg("projmatrix"),
                py::arg("tan_fovx"),
                py::arg("tan_fovy"),
                py::arg("image_height"),
                py::arg("image_width"),
                py::arg("dL_dout_color"),
                py::arg("dL_dout_depth"),
                py::arg("sh"),
                py::arg("degree"),
                py::arg("campos"),
                py::arg("geomBuffer"),
                py::arg("R"),
                py::arg("binningBuffer"),
                py::arg("imageBuffer"),
                py::arg("antialiasing"),
                py::arg("debug"),
                py::arg("axis")
        );
        m.def("mark_visible", &markVisible);
}