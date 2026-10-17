# Table S4. Input-Geometry Characteristics of Lung-Focused Preprocessing

All classifier inputs were bilinearly resized to 512 × 512. The study crop used the lung mask only to define a padded rectangular window (top 7%, bottom 10%, and left/right 5%); pixels within this window were not suppressed. Hard masking set non-lung pixels to zero while preserving the full radiograph extent. For the geometry measurements reported in this table, the additional control transformations (matched center crop, uniform 0%, 10%, and 20% margin crops, and Gaussian-blurred soft mask) were generated in memory. Separate classifier training for these control conditions was performed for the controlled classification experiments described in Section 3.4 and reported in Table 6 and Supplementary Table S9. Values are reported as median [IQR].

| Mask source | View | Metric | Median [IQR] | N |
| --- | --- | --- | ---: | ---: |
| MedSAM3 | AP | Original aspect ratio (W/H) | 0.906 [0.832–1.201] | 22,894 |
| MedSAM3 | AP | Study crop area / original area | 0.669 [0.588–0.755] | 22,894 |
| MedSAM3 | AP | Study crop aspect ratio (W/H) | 1.108 [1.003–1.220] | 22,894 |
| MedSAM3 | AP | Raw lung occupancy in 512² | 0.206 [0.173–0.245] | 22,894 |
| MedSAM3 | AP | Study crop lung occupancy in 512² | 0.313 [0.284–0.342] | 22,894 |
| MedSAM3 | AP | Lung-area magnification (crop / raw) | 1.495 [1.326–1.699] | 22,894 |
| MedSAM3 | AP | Hard-mask zero-pixel fraction in 512² | 0.794 [0.755–0.827] | 22,894 |
| MedSAM3 | AP | Matched center-crop area / original area | 0.669 [0.588–0.755] | 22,894 |
| MedSAM3 | AP | Matched center-crop lung occupancy in 512² | 0.305 [0.275–0.336] | 22,894 |
| MedSAM3 | AP | 0% margin crop area / original area | 0.470 [0.398–0.555] | 22,894 |
| MedSAM3 | AP | 0% margin lung occupancy in 512² | 0.448 [0.406–0.487] | 22,894 |
| MedSAM3 | AP | 10% margin crop area / original area | 0.749 [0.669–0.831] | 22,894 |
| MedSAM3 | AP | 10% margin lung occupancy in 512² | 0.278 [0.252–0.307] | 22,894 |
| MedSAM3 | AP | 20% margin crop area / original area | 0.936 [0.870–0.995] | 22,894 |
| MedSAM3 | AP | 20% margin lung occupancy in 512² | 0.224 [0.195–0.257] | 22,894 |
| MedSAM3 | AP | Soft-mask lung occupancy in 512² | 0.206 [0.173–0.245] | 22,894 |
| MedSAM3 | AP | Soft-mask exact-zero fraction in 512² | 0.478 [0.417–0.533] | 22,894 |
| MedSAM3 | AP | Soft-mask background fraction (1 − mean α) | 0.794 [0.755–0.827] | 22,894 |
| MedSAM3 | PA | Original aspect ratio (W/H) | 0.867 [0.832–1.000] | 18,609 |
| MedSAM3 | PA | Study crop area / original area | 0.748 [0.670–0.823] | 18,609 |
| MedSAM3 | PA | Study crop aspect ratio (W/H) | 1.032 [0.946–1.131] | 18,609 |
| MedSAM3 | PA | Raw lung occupancy in 512² | 0.254 [0.217–0.293] | 18,609 |
| MedSAM3 | PA | Study crop lung occupancy in 512² | 0.342 [0.316–0.369] | 18,609 |
| MedSAM3 | PA | Lung-area magnification (crop / raw) | 1.337 [1.216–1.494] | 18,609 |
| MedSAM3 | PA | Hard-mask zero-pixel fraction in 512² | 0.746 [0.707–0.783] | 18,609 |
| MedSAM3 | PA | Matched center-crop area / original area | 0.748 [0.670–0.823] | 18,609 |
| MedSAM3 | PA | Matched center-crop lung occupancy in 512² | 0.339 [0.312–0.366] | 18,609 |
| MedSAM3 | PA | 0% margin crop area / original area | 0.537 [0.464–0.612] | 18,609 |
| MedSAM3 | PA | 0% margin lung occupancy in 512² | 0.478 [0.444–0.511] | 18,609 |
| MedSAM3 | PA | 10% margin crop area / original area | 0.823 [0.751–0.891] | 18,609 |
| MedSAM3 | PA | 10% margin lung occupancy in 512² | 0.310 [0.283–0.338] | 18,609 |
| MedSAM3 | PA | 20% margin crop area / original area | 0.981 [0.925–1.000] | 18,609 |
| MedSAM3 | PA | 20% margin lung occupancy in 512² | 0.264 [0.232–0.299] | 18,609 |
| MedSAM3 | PA | Soft-mask lung occupancy in 512² | 0.254 [0.217–0.293] | 18,609 |
| MedSAM3 | PA | Soft-mask exact-zero fraction in 512² | 0.403 [0.347–0.461] | 18,609 |
| MedSAM3 | PA | Soft-mask background fraction (1 − mean α) | 0.746 [0.707–0.783] | 18,609 |
| CheXMask | AP | Original aspect ratio (W/H) | 0.906 [0.832–1.201] | 22,894 |
| CheXMask | AP | Study crop area / original area | 0.707 [0.637–0.779] | 22,894 |
| CheXMask | AP | Study crop aspect ratio (W/H) | 1.065 [0.965–1.175] | 22,894 |
| CheXMask | AP | Raw lung occupancy in 512² | 0.220 [0.188–0.259] | 22,894 |
| CheXMask | AP | Study crop lung occupancy in 512² | 0.315 [0.287–0.345] | 22,894 |
| CheXMask | AP | Lung-area magnification (crop / raw) | 1.415 [1.284–1.570] | 22,894 |
| CheXMask | AP | Hard-mask zero-pixel fraction in 512² | 0.780 [0.741–0.812] | 22,894 |
| CheXMask | AP | Matched center-crop area / original area | 0.707 [0.637–0.779] | 22,894 |
| CheXMask | AP | Matched center-crop lung occupancy in 512² | 0.310 [0.282–0.340] | 22,894 |
| CheXMask | AP | 0% margin crop area / original area | 0.504 [0.440–0.578] | 22,894 |
| CheXMask | AP | 0% margin lung occupancy in 512² | 0.444 [0.407–0.480] | 22,894 |
| CheXMask | AP | 10% margin crop area / original area | 0.789 [0.721–0.859] | 22,894 |
| CheXMask | AP | 10% margin lung occupancy in 512² | 0.281 [0.255–0.311] | 22,894 |
| CheXMask | AP | 20% margin crop area / original area | 0.968 [0.910–1.000] | 22,894 |
| CheXMask | AP | 20% margin lung occupancy in 512² | 0.232 [0.203–0.266] | 22,894 |
| CheXMask | AP | Soft-mask lung occupancy in 512² | 0.220 [0.188–0.259] | 22,894 |
| CheXMask | AP | Soft-mask exact-zero fraction in 512² | 0.450 [0.394–0.499] | 22,894 |
| CheXMask | AP | Soft-mask background fraction (1 − mean α) | 0.780 [0.741–0.812] | 22,894 |
| CheXMask | PA | Original aspect ratio (W/H) | 0.867 [0.832–1.000] | 18,609 |
| CheXMask | PA | Study crop area / original area | 0.759 [0.689–0.828] | 18,609 |
| CheXMask | PA | Study crop aspect ratio (W/H) | 1.009 [0.929–1.104] | 18,609 |
| CheXMask | PA | Raw lung occupancy in 512² | 0.257 [0.221–0.296] | 18,609 |
| CheXMask | PA | Study crop lung occupancy in 512² | 0.340 [0.313–0.368] | 18,609 |
| CheXMask | PA | Lung-area magnification (crop / raw) | 1.317 [1.208–1.452] | 18,609 |
| CheXMask | PA | Hard-mask zero-pixel fraction in 512² | 0.743 [0.704–0.779] | 18,609 |
| CheXMask | PA | Matched center-crop area / original area | 0.759 [0.689–0.828] | 18,609 |
| CheXMask | PA | Matched center-crop lung occupancy in 512² | 0.338 [0.311–0.366] | 18,609 |
| CheXMask | PA | 0% margin crop area / original area | 0.546 [0.482–0.616] | 18,609 |
| CheXMask | PA | 0% margin lung occupancy in 512² | 0.473 [0.440–0.505] | 18,609 |
| CheXMask | PA | 10% margin crop area / original area | 0.837 [0.773–0.898] | 18,609 |
| CheXMask | PA | 10% margin lung occupancy in 512² | 0.308 [0.281–0.337] | 18,609 |
| CheXMask | PA | 20% margin crop area / original area | 0.992 [0.942–1.000] | 18,609 |
| CheXMask | PA | 20% margin lung occupancy in 512² | 0.265 [0.233–0.300] | 18,609 |
| CheXMask | PA | Soft-mask lung occupancy in 512² | 0.257 [0.221–0.296] | 18,609 |
| CheXMask | PA | Soft-mask exact-zero fraction in 512² | 0.396 [0.343–0.449] | 18,609 |
| CheXMask | PA | Soft-mask background fraction (1 − mean α) | 0.743 [0.704–0.779] | 18,609 |

Study crop = current MedSAM3/CheXMask crop used in the 70 trained configurations. Matched center crop uses the same rectangle size placed at the image center. Soft mask uses a Gaussian blur with radius 2% of the shorter image side.

**Raw-input reference.** Raw is the uncropped full-image condition. Its retained area is 1.0 and its lung-area magnification is 1.0 by definition. Because lung occupancy requires a reference mask, raw occupancy is reported separately under the MedSAM3 and CheXMask rows (`Raw lung occupancy in 512²`). Crop aspect ratio and hard-mask zero fraction are not properties of the unmodified raw image.
