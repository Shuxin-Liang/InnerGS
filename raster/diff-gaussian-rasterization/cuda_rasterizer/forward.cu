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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ----------------------------------------------------------
// 辅助函数 (无需修改)
// ----------------------------------------------------------
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// 使用条件概率公式进行简化，符合我们的模型，保持不变
	glm::mat3 cov = glm::mat3(0.0f);
	float cov3d_5_safe = cov3D[5] + 1e-6;
	cov[0][0] = cov3D[0] - (cov3D[2] * cov3D[2] / cov3d_5_safe);
	cov[0][1] = cov3D[1] - (cov3D[2] * cov3D[4] / cov3d_5_safe);
	cov[1][0] = cov[0][1];
	cov[1][1] = cov3D[3] - (cov3D[4] * cov3D[4] / cov3d_5_safe);

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;
	glm::mat3 Sigma = glm::transpose(M) * M;

	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// ----------------------------------------------------------
// 1. preprocessCUDA 核函数 (Z轴切片简化版)
// ----------------------------------------------------------
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P) 
		return;

	radii[idx] = 0;
	tiles_touched[idx] = 0;

	if (cov3D_precomp == nullptr)
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
	}
	const float* cov3D = (cov3D_precomp != nullptr) ? cov3D_precomp + idx * 6 : cov3Ds + idx * 6;

	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	
	// Compute 2D screen-space covariance matrix
	//原始代码中computeCov2D 输出的协方差矩阵，是在像素空间 (pixel space)。
    // =================== 协方差转换的关键修改区域 ===================

    // 1. 首先，计算归一化空间下的2D协方差, condiitonal 2D, original space
    float3 cov_norm = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

    // 2. 将其重构为2x2矩阵
    glm::mat2 cov_norm_mat = glm::mat2(cov_norm.x, cov_norm.y, cov_norm.y, cov_norm.z);

    // 3. 定义从归一化空间到像素空间的缩放矩阵
    //glm::mat2 scale_mat = glm::mat2(W - 1, 0.0f, 0.0f, H - 1);
	glm::mat2 scale_mat = glm::mat2(W / 2.0f, 0.0f, 0.0f, H / 2.0f);
    
    // 4. 执行变换，得到像素空间下的协方差矩阵: Sigma_pix = S * Sigma_norm * S
    glm::mat2 cov_pixel_mat = scale_mat * cov_norm_mat * scale_mat;
    
    // -- START: 在此处添加正则化项 --
    cov_pixel_mat[0][0] += 0.1f;
    cov_pixel_mat[1][1] += 0.1f;
    // -- END: 在此处添加正则化项 --

    // 5. 提取变换后的协方差分量，用于后续计算
    float3 cov = {cov_pixel_mat[0][0], cov_pixel_mat[0][1], cov_pixel_mat[1][1]};
    
    // ===================完成2D屏幕空间协方差矩阵计算===========================================

	// 后续的计算现在使用的是像素空间下的协方差
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f) 
		return;
    float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

    // 均值也被正确地转换到像素空间
    // 将[-1, 1]的坐标范围映射到[-0.5, W-0.5]和[-0.5, H-0.5]
    // 在此处没有使用ndc2Pix函数，而是直接计算, mu_u->mu_u',mu_v->mu_v'
	// =================== 此处将均值转换到像素空间 (已修正) ===================
    // 1. 获取3D均值和协方差分量
    float mu_x = p_orig.x;
    float mu_y = p_orig.y;
    float mu_z = p_orig.z;
    float sigma_xz = cov3D[2];
    float sigma_yz = cov3D[4];
    float sigma_zz = cov3D[5];

    // 2. 计算条件均值 (在世界坐标系)
    // 这里的 cam_pos->z 是切片的位置 t
    float t_minus_mu_z = cam_pos->z - mu_z;
    float inv_sigma_zz = 1.0f / (sigma_zz + 1e-8f);
    float mu_u_world = mu_x + sigma_xz * t_minus_mu_z * inv_sigma_zz;
    float mu_v_world = mu_y + sigma_yz * t_minus_mu_z * inv_sigma_zz;

    // 3. 将世界坐标系下的条件均值转换到像素空间
    // 我们假设世界坐标在 [-1, 1] 范围内，与NDC对齐
    float2 point_image = { ((mu_u_world + 1.0f) * W - 1.0f) * 0.5f, 
                           ((mu_v_world + 1.0f) * H - 1.0f) * 0.5f };
    // ===================完成2D屏幕空间均值计算======================
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0) 
		return;

	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

    // =================== 深度计算逻辑 (硬编码Z轴) ===================
    // 计算高斯中心到Z轴切片平面的距离，并存入depths数组用于排序。
    //const float* means3D = orig_points;
    //float mu_z = means3D[idx * 3 + 2]; // 直接取Z轴坐标 (索引2) //这里不需要重复计算了   
    // 高斯中心到Z轴切片平面的距离
    //float dist = abs(cam_pos->z - mu_z);

    // 为了实现降序排序（距离远的先渲染），我们存入负值。
    // 此处重新定义了depth
    float dist = abs(cam_pos->z - p_orig.z); 
	depths[idx] = -dist;    
    // ===============================================================
	
	radii[idx] = my_radius;
    // (mu_u',mu_v') //以下全部是在像素空间
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// ----------------------------------------------------------
// 2. renderCUDA 核函数 (Z轴切片简化版)
// ----------------------------------------------------------
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
    // --- 用于G1D计算的参数 ---
    const float* __restrict__ means3D,//added 这个真的需要添加吗
    const float* __restrict__ cov3Ds,//added
	const float* __restrict__ depths,//added
    // --- 结束 ---
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
	
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;
	
	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	
	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	// 新增的共享内存数组
	__shared__ float collected_sigma_zz[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	
	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[NUM_CHANNELS] = { 0 };
	
	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{		
	// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;
		
		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			// 添加范围检查，确保coll_id在有效范围内
			// 注意：这里需要知道P（高斯数量）的值，但renderCUDA函数没有P参数
			// 我们假设means3D指向的是P个高斯的数据，所以用一个大的上限检查
			// 或者，如果coll_id是负数或非常大，我们就跳过它
			if (coll_id >= 0) {
				collected_id[block.thread_rank()] = coll_id;
				collected_xy[block.thread_rank()] = points_xy_image[coll_id];
				collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
				// --- 在这里从全局内存读取，存入共享内存 ---
				collected_sigma_zz[block.thread_rank()] = cov3Ds[coll_id * 6 + 5];
				collected_depth[block.thread_rank()] = depths[coll_id];
			} else {
				// 如果coll_id无效，设置collected_id为-1，后续会被跳过
				collected_id[block.thread_rank()] = -1;
			}
		}
		else
		{
			// 如果超出范围，设置collected_id为-1
			collected_id[block.thread_rank()] = -1;
		}
		block.sync();
		
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// 检查collected_id是否有效
			if (collected_id[j] < 0)
				continue;
				
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			//const int global_id = collected_id[j];//不需要在这里

			// 此处是(mu_u',mu_v')-像素点坐标
			//collected_xy[block.thread_rank()] = points_xy_image[coll_id];此处是转换到像素空间的坐标
			float2 xy = collected_xy[j];
            float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];

			// =================== 最终的Alpha计算逻辑 (Z轴简化版) ===================
			// 步骤 1: G2D 计算 (保持不变)
			float power2D = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power2D > 0.0f) 
				continue;
			float G2D = expf(power2D);

			// 步骤 2: G1D 计算 (使用 preprocess 中计算的 depths)
			//float sigma_zz = cov3Ds[global_id * 6 + 5]; // 3D协方差的第6个元素是zz
			//float diff = depths[global_id];
			float sigma_zz = collected_sigma_zz[j];
			float diff = collected_depth[j];
			float power1D = -0.5f * (diff * diff) / (sigma_zz + 1e-8f);
			float G1D = expf(power1D);
			
			// 步骤 3: 最终 Alpha 计算
			const float opacity = con_o.w;
			float alpha = min(0.99f, opacity * G1D * G2D);
			// ====================================================================

			if (alpha < 1.0f / 255.0f) 
				continue;
			float test_T = T * (1.f - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < NUM_CHANNELS; ch++)
				C[ch] += features[collected_id[j] * NUM_CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}
	
	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < NUM_CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// ----------------------------------------------------------
// 3. FORWARD 命名空间中的包装函数 (Z轴切片简化版)
// ----------------------------------------------------------

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS><<<(P + 255) / 256, 256>>>(
		P, D, M,
		means3D,
		scales, 
		scale_modifier, 
		rotations, 
		opacities, 
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		focal_x, focal_y, 
		tan_fovx, tan_fovy,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered);
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* points_xy_image,
	const float* features,
	const float4* conic_opacity,
	const float* means3D,//added
	const float* cov3Ds,//added
	const float* depths,//added
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color)
{
	renderCUDA<NUM_CHANNELS><<<grid, block>>>(
		ranges,
		point_list,
		W, H,
		points_xy_image,
		features,
		conic_opacity,
		means3D,//added
		cov3Ds,//added
		depths,//added
		final_T,
		n_contrib,
		bg_color,
		out_color);
}