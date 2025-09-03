#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// ==========================================================
// Additional helper structures and functions (same as in forward.cu)
// ==========================================================
struct float3x3 { float m[3][3]; };

// Matrix-vector multiplication
__device__ __forceinline__ float3 mat_vec_mult(const float3x3& M, const float3& v)
{
    return { M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z,
             M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z,
             M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z };
}

// Vector dot product
__device__ __forceinline__ float dot(const float3& v1, const float3& v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// 3x3 matrix inversion (analytical method)
__device__ __forceinline__ float3x3 invert_3x3(const float* cov3D)
{
    float3x3 inv;
    float a = cov3D[0], b = cov3D[1], c = cov3D[2];
    float d = cov3D[3], e = cov3D[4], f = cov3D[5];

    float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    float inv_det = 1.0f / (det + 1e-12f);

    inv.m[0][0] = (d * f - e * e) * inv_det;
    inv.m[0][1] = (c * e - b * f) * inv_det;
    inv.m[0][2] = (b * e - c * d) * inv_det;
    inv.m[1][0] = (c * e - b * f) * inv_det;
    inv.m[1][1] = (a * f - c * c) * inv_det;
    inv.m[1][2] = (b * c - a * e) * inv_det;
    inv.m[2][0] = (b * e - c * d) * inv_det;
    inv.m[2][1] = (b * c - a * e) * inv_det;
    inv.m[2][2] = (a * d - b * b) * inv_det;

    return inv;
}

// New: Calculate 3D world coordinates based on axis direction
__device__ __forceinline__ float3 compute_3d_point(float x_world, float y_world, const glm::vec3& cam_pos, int axis)
{
    if (axis == 0) { // X-axis slice: YZ plane
        return {cam_pos.x, x_world, y_world};
    }
    else if (axis == 1) { // Y-axis slice: XZ plane
        return {x_world, cam_pos.y, y_world};
    }
    else { // Z-axis slice: XY plane
        return {x_world, y_world, cam_pos.z};
    }
}

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

// computeCov2DCUDA function has been deleted, as it is no longer needed when using G3D

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

// preprocessCUDA function has been deleted, as it is no longer needed when using G3D

template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	int P,
	const float* __restrict__ bg_color,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
    // --- New inputs (for G3D calculation) ---
    const float3* __restrict__ means3D,
    const float* __restrict__ cov3Ds,
    const glm::vec3* __restrict__ cam_pos,
    int axis,  // New axis direction parameter
    // --- Output gradients ---
    float3* __restrict__ dL_dmeans,
    float* __restrict__ dL_dcov3D,
    float* __restrict__ dL_dopacity,
    float* __restrict__ dL_dcolors
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
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	// New shared memory arrays (for G3D calculation)
	__shared__ float3 collected_means3D[BLOCK_SIZE];
	__shared__ float collected_cov3D[BLOCK_SIZE * 6];
	
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
			if (coll_id >= 0 && coll_id < P) {
				collected_id[block.thread_rank()] = coll_id;
				collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
				for (int ch = 0; ch < C; ch++)
					collected_colors[ch * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + ch];
				// --- Prefetch 3D Gaussian parameters from global memory here ---
				collected_means3D[block.thread_rank()] = means3D[coll_id];
				for (int j = 0; j < 6; j++)
					collected_cov3D[block.thread_rank() * 6 + j] = cov3Ds[coll_id * 6 + j];
			} else {
				collected_id[block.thread_rank()] = P;
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

			// =================== G3D calculation logic (multi-axis support) ===================
			// 1. Convert pixel coordinates back to points on slice plane in world coordinate system
			float x_world = (2.0f * pixf.x + 1.0f) / W - 1.0f;
			float y_world = (2.0f * pixf.y + 1.0f) / H - 1.0f;
			float3 p_slice = compute_3d_point(x_world, y_world, *cam_pos, axis);

			// 2. Get Gaussian parameters from shared memory
			const float3 mu = collected_means3D[j];
			const float* cov3D = &collected_cov3D[j * 6];

			// 3. Calculate 3D deviation vector
			float3 d_3D = {p_slice.x - mu.x, p_slice.y - mu.y, p_slice.z - mu.z};

			// 4. Calculate 3D Gaussian power
			float3x3 inv_cov = invert_3x3(cov3D);
			float3 v = mat_vec_mult(inv_cov, d_3D);
			float power = -0.5f * dot(d_3D, v);

			if (power > 0.0f)
				continue;

			// 5. Calculate final Alpha
			float G3D = expf(power);
			const float opacity = collected_conic_opacity[j].w;
			const float alpha = min(0.99f, opacity * G3D);

			if (alpha < 1.0f / 255.0f)
				continue;

			// Calculate dL_dalpha (this part of logic unchanged)
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

			// =================== G3D gradient calculation ===================
			// Calculate dL/d(power)
			float dL_dG3D = dL_dalpha * opacity;
			float dL_dpower = dL_dG3D * G3D;

			// Calculate and accumulate gradients with respect to mean
			// dL/dμ = dL/dpower * ∂power/∂μ = dL/dpower * Σ^(-1) * (x - μ)
			float3 dL_dmean_pixel = {dL_dpower * v.x, dL_dpower * v.y, dL_dpower * v.z};
			atomicAdd(&dL_dmeans[global_id].x, dL_dmean_pixel.x);
			atomicAdd(&dL_dmeans[global_id].y, dL_dmean_pixel.y);
			atomicAdd(&dL_dmeans[global_id].z, dL_dmean_pixel.z);

			// Calculate and accumulate gradients with respect to covariance
			// dL/dΣ = 0.5 * dL/dpower * Σ^(-1) * (x-μ)(x-μ)^T * Σ^(-1)
			float dL_dSigma_factor = 0.5f * dL_dpower;
			atomicAdd(&dL_dcov3D[global_id * 6 + 0], dL_dSigma_factor * v.x * v.x); // xx
			atomicAdd(&dL_dcov3D[global_id * 6 + 1], dL_dSigma_factor * 2.f * v.x * v.y); // xy
			atomicAdd(&dL_dcov3D[global_id * 6 + 2], dL_dSigma_factor * 2.f * v.x * v.z); // xz
			atomicAdd(&dL_dcov3D[global_id * 6 + 3], dL_dSigma_factor * v.y * v.y); // yy
			atomicAdd(&dL_dcov3D[global_id * 6 + 4], dL_dSigma_factor * 2.f * v.y * v.z); // yz
			atomicAdd(&dL_dcov3D[global_id * 6 + 5], dL_dSigma_factor * v.z * v.z); // zz

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G3D * dL_dalpha);
		}
	}
}

__global__ void preprocessCUDA(
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
    const float* dL_dcolors,
    glm::vec3* dL_dscales,
    glm::vec4* dL_drots,
    float* dL_dshs)
{
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P) return;

    if (radii[idx] <= 0) return;

    // Propagate gradients to SHs
    if (shs != nullptr)
    {
        computeColorFromSH(
            idx, D, M,
            (const glm::vec3*)means3D,
            *campos,
            shs,
            clamped,
            (const glm::vec3*)dL_dcolors,
            (glm::vec3*)dL_dmeans,
            (glm::vec3*)dL_dshs
        );
    }
    
    // Propagate gradients to scales and rotations
    if (scales != nullptr)
    {
        computeCov3D(
            idx,
            scales[idx],
            scale_modifier,
            rotations[idx],
            dL_dcov3D,
            dL_dscales,
            dL_drots
        );
    }
}

void BACKWARD::preprocess(
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
    float* dL_dshs)
{
    preprocessCUDA << <(P + 255) / 256, 256 >> > (
        P, D, M,
        means3D,
        radii,
        shs,
        clamped,
        scales,
        rotations,
        scale_modifier,
        campos,
        dL_dcov3D,
        dL_dmeans,
        dL_dcolors,
        dL_dscales,
        dL_drots,
        dL_dshs);
}

void BACKWARD::render(
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
)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		P,
		bg_color,
		conic_opacity,
		colors,
		final_Ts,
		n_contrib,
		dL_dpixels,
        // --- Pass new parameters ---
        means3D,
        cov3Ds,
        campos,
        axis,  // Pass axis direction parameter
        // ---
		dL_dmeans,
		dL_dcov3D,
		dL_dopacity,
		dL_dcolors
		);
}