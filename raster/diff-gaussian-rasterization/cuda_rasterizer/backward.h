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

#pragma once

#include <iostream>
#include "config.h"
#include <glm/glm.hpp>

namespace BACKWARD
{
	void preprocess(
		int P, int D, int M, int W, int H, //  <-- 确保此行有 W 和 H
		const float3* means3D,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconic,
		const float* dL_dG1D,
		const float* depths,
		glm::vec3* dL_dmean3D,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);

	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int P,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const glm::vec3* campos,
		const float3* means3D,
		const float* depths,
		const float* cov3Ds,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_dG1Ds);
}
