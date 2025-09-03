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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
    void render(
        const dim3 grid, const dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        int P,
        const float* bg_color,
        const float4* conic_opacity,
        const float* colors,
        const float* final_Ts,
        const uint32_t* n_contrib,
        const float* dL_dpixels,
        // --- New input parameters ---
        const float3* means3D,
        const float* cov3Ds,
        const glm::vec3* campos,
        int axis,  // New axis direction parameter
        // --- Output parameters ---	
        float3* dL_dmeans,
        float* dL_dcov3D,
        float* dL_dopacity,
        float* dL_dcolors
    );

	void preprocess(
		int P, int D, int M,
		const float3* means3D,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const glm::vec3* campos,
		const float* dL_dcov3D,
		const float3* dL_dmeans,
		float* dL_dcolors,
		glm::vec3* dL_dscales,
		glm::vec4* dL_drots,
		float* dL_dshs);
}

#endif
