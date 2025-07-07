#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

__global__ void computeCov2DCUDA(int P, int W, int H,
	const int* radii,
	const float* cov3Ds,
	const float* dL_dconics,
	float* dL_dcov_from_cov2D)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Σ_i^3D
	const float* cov3D = cov3Ds + 6 * idx;

	// µ_i^3D
	//float3 mean = means[idx];

	// ∂L/∂(Σ2D')^-1, pixel space
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	
	// Σ2D, normalized space
	glm::mat2 cov_norm = glm::mat2(0.0f);
	float cov3d_5_safe = cov3D[5] + 1e-6;
	cov_norm[0][0] = cov3D[0] - (cov3D[2] * cov3D[2] / cov3d_5_safe);
	cov_norm[0][1] = cov3D[1] - (cov3D[2] * cov3D[4] / cov3d_5_safe);
	cov_norm[1][0] = cov_norm[0][1]; 
	cov_norm[1][1] = cov3D[3] - (cov3D[4] * cov3D[4] / cov3d_5_safe);
	
	glm::mat2 scale_mat = glm::mat2(W / 2.0f, 0.0f, 0.0f, H / 2.0f);
	// Σ2D' (pixel space)
	glm::mat2 cov2D = scale_mat * cov_norm * scale_mat;
	
	//Σ2D'是在pixel space加0.3 (pixel space)
	float a = cov2D[0][0] += 0.1f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.1f;
	// det(Σ2D) (pixel space)
	float denom = a * c - b * b;
	float dL_da_pix = 0, dL_db_pix = 0, dL_dc_pix = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
	// ∂(Σ2D')^(-1)/∂(Σ2D')  (pixel space)
		dL_da_pix = denom2inv * (-(c * c) * dL_dconic.x + (2.f * b * c) * dL_dconic.y - (b * b) * dL_dconic.z);
		dL_dc_pix = denom2inv * (-(b * b) * dL_dconic.x + (2.f * a * b) * dL_dconic.y - (a * a) * dL_dconic.z);
		dL_db_pix = denom2inv * (2.f * b * c * dL_dconic.x - 2.f * (a * c + b * b) * dL_dconic.y + 2.f * a * b * dL_dconic.z);

	// ∂(Σ2D')/∂(Σ2D)  (pixel space to normalized space)
		float dL_da_norm = W * W * dL_da_pix / 4.0f;
		float dL_db_norm = W * H * dL_db_pix / 4.0f;
		float dL_dc_norm = H * H * dL_dc_pix / 4.0f;

	// ∂(Σ2D)/∂(Σ3D) (normalized space)
		float cov3d_5_safe_sq = cov3d_5_safe * cov3d_5_safe;
		dL_dcov_from_cov2D[6 * idx + 0] = dL_da_norm;
		dL_dcov_from_cov2D[6 * idx + 1] = 2.f * dL_db_norm;
		dL_dcov_from_cov2D[6 * idx + 2] = dL_da_norm * (-2.f * cov3D[2] / cov3d_5_safe) + 2.f * dL_db_norm * (-cov3D[4] / cov3d_5_safe);
		dL_dcov_from_cov2D[6 * idx + 3] = dL_dc_norm;
		dL_dcov_from_cov2D[6 * idx + 4] = 2.f * dL_db_norm * (-cov3D[2] / cov3d_5_safe) + dL_dc_norm * (-2.f * cov3D[4] / cov3d_5_safe);
		dL_dcov_from_cov2D[6 * idx + 5] = dL_da_norm * (cov3D[2] * cov3D[2] / cov3d_5_safe_sq) + 2 * dL_db_norm * (cov3D[2] * cov3D[4] / cov3d_5_safe_sq) + dL_dc_norm * (cov3D[4] * cov3D[4] / cov3d_5_safe_sq);
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov_from_cov2D[6 * idx + i] = 0;
	}
}

__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

template<int C>
__global__ void preprocessCUDA(
    int P, int D, int M, 
    const float3* means,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    const glm::vec3* campos,
	// --- Input Gradients & Data ---
    const float3* dL_dmean2D,
	const float* dL_dG1D,      // [必需] 后向传播的 render 函数计算出的新梯度 dL/d(G_1D)
    const float* depths,
	const float* dL_dcov_from_cov2D, // <<< [INPUT] The pre-computed gradient from the temporary buffer.
    // --- Output Gradients ---
	glm::vec3* dL_dmeans,
    float* dL_dcolor,
    float* dL_dcov3D,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P || !(radii[idx] > 0))
        return;

    //--------------------------------------------------------------------
    // 步骤1: 整合对 3D 协方差 (dL_dcov3D) 的所有梯度
    //--------------------------------------------------------------------
    const float* cov3D = cov3Ds + 6 * idx;
    float sigma_xz = cov3D[2];
    float sigma_yz = cov3D[4];
    float sigma_zz = cov3D[5];

    // --- 步骤 1.1: 从路径 L -> Σ_2D -> Σ_3D 初始化梯度 ---
    for(int i = 0; i < 6; ++i) {
        dL_dcov3D[6 * idx + i] = dL_dcov_from_cov2D[6 * idx + i];
    }

    // 准备计算后续梯度所需的变量
    float mu_z = means[idx].z;
    float t_minus_mu_z = campos->z - mu_z; // <--- depth = t - mu_z
    float inv_sigma_zz = 1.0f / (sigma_zz + 1e-8f);

    // --- 步骤 1.2: 累加路径 L -> μ_2D -> Σ_3D 的梯度 ---
    //[cite_start]// 公式: d(mu_u)/d(sigma_xz) = (t - mu_z) / sigma_zz [cite: 119]
    atomicAdd(&dL_dcov3D[6 * idx + 2], dL_dmean2D[idx].x * t_minus_mu_z * inv_sigma_zz);
    //[cite_start]// 公式: d(mu_v)/d(sigma_yz) = (t - mu_z) / sigma_zz [cite: 121]
    atomicAdd(&dL_dcov3D[6 * idx + 4], dL_dmean2D[idx].y * t_minus_mu_z * inv_sigma_zz);
    //[cite_start]// 公式: d(mu_u)/d(sigma_zz) = -sigma_xz * (t - mu_z) / (sigma_zz^2) [cite: 122]
    //[cite_start]// 公式: d(mu_v)/d(sigma_zz) = -sigma_yz * (t - mu_z) / (sigma_zz^2) [cite: 122]
    float dmu_u_dsigma_zz = -sigma_xz * t_minus_mu_z * inv_sigma_zz * inv_sigma_zz;
    float dmu_v_dsigma_zz = -sigma_yz * t_minus_mu_z * inv_sigma_zz * inv_sigma_zz;
    atomicAdd(&dL_dcov3D[6 * idx + 5], dL_dmean2D[idx].x * dmu_u_dsigma_zz + dL_dmean2D[idx].y * dmu_v_dsigma_zz);

    // --- 步骤 1.3: 累加路径 L -> G_1D -> Σ_3D 的梯度 ---
    float G1D_i = expf(-0.5f * (t_minus_mu_z * t_minus_mu_z) * inv_sigma_zz);
    //[cite_start]// 公式: d(G_1D)/d(sigma_zz) = (1/2) * G_1D * ((mu_z - t) / sigma_zz)^2 [cite: 83]
    float dG1D_dsigma_zz = 0.5f * G1D_i * (t_minus_mu_z * t_minus_mu_z) * (inv_sigma_zz * inv_sigma_zz);
    atomicAdd(&dL_dcov3D[6 * idx + 5], dL_dG1D[idx] * dG1D_dsigma_zz);

    //--------------------------------------------------------------------
    // 步骤 2: 计算对 3D 均值 (dL_dmeans) 的最终梯度
    //--------------------------------------------------------------------
    glm::vec3 dL_dmean;

    // --- 步骤 2.1: 计算来自路径 L -> μ_2D -> μ_3D 的梯度 ---
	// d(mu_x)/d(mu_u)= 1
	// d(mu_y)/d(mu_v)= 1
    dL_dmean.x = dL_dmean2D[idx].x;
    dL_dmean.y = dL_dmean2D[idx].y;
    //[cite_start]// 公式: d(mu_u)/d(mu_z) = -sigma_xz / sigma_zz, d(mu_v)/d(mu_z) = -sigma_yz / sigma_zz [cite: 90]
    dL_dmean.z = dL_dmean2D[idx].x * (-sigma_xz * inv_sigma_zz) + dL_dmean2D[idx].y * (-sigma_yz * inv_sigma_zz);

    // --- 步骤 2.2: 累加来自路径 L -> G_1D -> μ_3D 的梯度 ---
    //[cite_start]// 公式: d(G_1D)/d(mu_z) = -G_1D * (mu_z - t) / sigma_zz [cite: 83]
    float dG1D_dmu_z = G1D_i * t_minus_mu_z * inv_sigma_zz;
    dL_dmean.z += dL_dG1D[idx] * dG1D_dmu_z;

    // 将计算出的梯度累加到总梯度上
    dL_dmeans[idx] += dL_dmean;

    //--------------------------------------------------------------------
    // 步骤 3: 将总梯度反向传播到最终参数 (SH, scale, rotation)
    //--------------------------------------------------------------------
    // --- 步骤 3.1: 传播到球谐函数 (SH) 和 3D 均值 (颜色部分) ---
    if (shs)
        computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

    // --- 步骤 3.2: 传播到缩放 (scale) 和旋转 (rotation) ---
    if (scales)
        computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	int P,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
    // --- 新增输入 (方案A: 在核函数内部计算 G_1D) ---
	const glm::vec3* __restrict__ cam_pos, // <-- 添加此行
    const float3* __restrict__ means,
    const float* __restrict__ depths,
    const float* __restrict__ cov3Ds,
    // --- 输出 ---
    float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
    // --- 新增输出 ---
    float* __restrict__ dL_dG1Ds
)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	//增加两个共享内存数组
	__shared__ float collected_mu_z[BLOCK_SIZE];
	__shared__ float collected_sigma_zz[BLOCK_SIZE];
	
	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			// 添加范围检查，确保coll_id在有效范围内
			if (coll_id >= 0 && coll_id < P) {
				collected_id[block.thread_rank()] = coll_id;
				collected_xy[block.thread_rank()] = points_xy_image[coll_id];
				collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
				for (int ch = 0; ch < C; ch++)
					collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ch];
				// --- 在这里从全局内存预取数据 ---
				collected_mu_z[block.thread_rank()] = means[coll_id].z;
				collected_sigma_zz[block.thread_rank()] = cov3Ds[6 * coll_id + 5];
			} else {
				// 如果coll_id超出范围，设置为无效值
				collected_id[block.thread_rank()] = P; // 设置为P，后续会被跳过
			}
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;
			
			const int global_id = collected_id[j];
			if (global_id >= P)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];

            // 1. 计算 power2D 和 G2D (图像平面)
			const float power2D = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power2D > 0.0f)
				continue;
            const float G2D = expf(power2D);

            // 2. 计算 power1D 和 G1D (深度轴)
            //const float mu_z = means[global_id].z;
            //const float sigma_zz = cov3Ds[6 * global_id + 5];
            // 从共享内存中读取，而不是全局内存
            const float mu_z = collected_mu_z[j];
            const float sigma_zz = collected_sigma_zz[j];

			const float diff = cam_pos->z - mu_z;
            const float power1D = -0.5f * (diff * diff) / (sigma_zz + 1e-8f);
            const float G1D = expf(power1D);

            // 3. 计算最终的 alpha
            const float opacity = con_o.w;
			const float alpha = min(0.99f, opacity * G1D * G2D);

			if (alpha < 1.0f / 255.0f)
				continue;

            // 计算 dL_dalpha (这部分逻辑不变)
			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.				
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int ch = 0; ch < C; ch++)
				bg_dot_dpixel += bg_color[ch] * dL_dpixel[ch];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

            // 根据新的 alpha 定义分解梯度
            const float dL_dG1D = dL_dalpha * opacity * G2D;
            const float dL_dG2D = dL_dalpha * opacity * G1D;
            atomicAdd(&(dL_dG1Ds[global_id]), dL_dG1D);

			// 后续计算使用 dL_dG2D
			const float gdx = G2D * d.x;
			const float gdy = G2D * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
			
			// Update gradients w.r.t. 2D mean position of the Gaussian (to normalized space)
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG2D * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG2D * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG2D);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * (gdx * d.y + gdy * d.x) * dL_dG2D);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG2D);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G1D * G2D * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
    int P, int D, int M, int W, int H,
    const float3* means3D,
    const int* radii,
    const float* shs,
    const bool* clamped,
    const glm::vec3* scales,
    const glm::vec4* rotations,
    const float scale_modifier,
    const float* cov3Ds,
    //const float* viewmatrix,
    const glm::vec3* campos,
	//来自render的梯度输入
    const float3* dL_dmean2D,
    const float* dL_dconic,
	//增加输入
    const float* dL_dG1D,
	const float* depths,
	//输出
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
    float* dL_dcov3D,
    float* dL_dsh,
    glm::vec3* dL_dscale,
    glm::vec4* dL_drot)
{
    // 为 L -> Σ_2D -> Σ_3D 路径的梯度创建一个临时的GPU内存Buffer
    size_t dL_dcov_buffer_size = 6 * P * sizeof(float);
    float* dL_dcov_buffer;
    cudaMalloc(&dL_dcov_buffer, dL_dcov_buffer_size);
    cudaMemset(dL_dcov_buffer, 0, dL_dcov_buffer_size); // 最好清零
    // 第一步：计算部分梯度，并存入临时Buffer
    computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
        P, 
        W, H,
        radii, 
        cov3Ds, 
        dL_dconic, 
        dL_dcov_buffer // 将结果写入临时Buffer
        );

	// 第二步：调用主预处理核函数，它会读取临时Buffer中的梯度，
	// 并累加上其他路径的梯度，最终结果写入 dL_dcov3D
    preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
        P, D, M,
        (float3*)means3D,
        radii, 
        shs, 
        clamped,
        scales, 
        rotations, 
        scale_modifier,
        cov3Ds,
        campos,
        dL_dmean2D,
        dL_dG1D,
        depths,		
        dL_dcov_buffer, // Pass the temporary buffer as input
		// Outputs
		dL_dmean3D,
		dL_dcolor, 
        dL_dcov3D,      // 这是最终的梯度输出
        dL_dsh, 
        dL_dscale, 
        dL_drot
    );

    // 释放临时Buffer
    cudaFree(dL_dcov_buffer);
}

void BACKWARD::render(
    const dim3 grid, const dim3 block,
    const uint2* ranges,
    const uint32_t* point_list,
    int W, int H,
    int P,
    const float* bg_color,
    const float2* means2D,
    const float4* conic_opacity,
    const float* colors,
    const float* final_Ts,
    const uint32_t* n_contrib,
    const float* dL_dpixels,
	//const float* dL_invdepths,//这是什么
    // --- 新增输入参数 ---
	const glm::vec3* campos,
    const float3* means3D,
    const float* depths,
	const float* cov3Ds,
    // --- 输出参数 ---	
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	//float* dL_dinvdepths,//这是什么
	// --- 新增输出参数 ---
	float* dL_dG1Ds
)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		P,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
        // --- 传递新增参数 ---
        campos,
        means3D,
        depths,
        cov3Ds,
        // ---
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
        dL_dG1Ds
		);
}