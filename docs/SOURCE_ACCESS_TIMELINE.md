# Source access and project timeline

This records the project work and correspondence relevant to the official data
sources. Dates refer to November 2026. Private download links, credentials,
message identifiers, and source images are deliberately excluded from the
public repository. The [implementation dossier](implementation_dossier.pdf)
contains the illustrated architecture and evidence map.

| Date | Event | Evidence boundary |
| --- | --- | --- |
| 3 Nov | Requested clarification from EyePACS about internship research use, model training, sharing trained weights, and including original fundus image examples in a reviewer report. | The request itself grants no permission. |
| 3 Nov | Submitted the official ADCIS Messidor and Messidor-2 access forms. When the personal download messages did not arrive, contacted ADCIS support for an alternative route. | A successful form submission was not treated as a downloaded dataset. |
| 4 Nov | EyePACS contact directed the project to the official Kaggle competition release and requested credit to EyePACS, Inc. and Cuadros and Bresnick (2009) in reports or publications. | The response did not expressly resolve all requested post-competition use, weight-sharing, and image-bearing report scopes. |
| 4 Nov | Downloaded and verified all five original EyePACS training archive parts and the label archive in private storage. A separate Kaggle preparation job decoded, CRC-checked, hashed, and reconciled all 35,126 labeled JPEGs. | The grouped split has 24,590 training, 5,268 validation, and 5,268 sealed test images. This is source preparation, not classifier training. |
| 4 Nov | Ran two full-model optimizer steps on real EyePACS images using an earlier BF16/parallel-scan source revision. Built and compared a pinned low-level CUDA scan separately. | These runs qualify limited numerical and performance behavior; neither is a completed epoch, trained checkpoint, or classifier result. |
| 4 Nov | ADCIS technical support responded to the failed personal-link route with restricted alternative access to official original Messidor and Messidor-2 files. | ADCIS requested that the private access link not be shared. Only official data was used; no mirror or substitute dataset was introduced. |
| 4 Nov | Downloaded and integrity-checked the original Messidor archives and annotation workbooks, then extracted 1,200 TIFF images and the 12 canonical workbooks into private storage. The preparer decoded, hashed and reconciled every TIFF; its schema-valid manifest records four grade corrections and 13 published duplicate exclusions, leaving 1,187 eligible external images. | The public repository contains neither source images nor the restricted access method. The manifest is data preparation, not an external model result. |
| 4 Nov | Downloaded all four official Messidor-2 image archive parts through the same restricted ADCIS route. Reconstructed the ZIP, checked every ZIP member CRC, then extracted, decoded and hashed all 1,748 images; all byte hashes are unique. | The image ZIP contains no pairing spreadsheet. A five-grade Messidor-2 evaluation requires a separately verified examination mapping and the independent Google Brain grades. ADCIS says 1,058 images come from original Messidor, so this cohort overlaps the required external test. |
| 4 Nov | Checked the model block against Algorithm 1 of the Vision Mamba paper and changed directional combination from an average to the paper's sum before the shared gate and output projection. | The prior two-step GPU qualification predates this correction; the current exact source requires its own numerical and full-model run check. |

## Source roles

| Source | Purpose in this project | Public reference |
| --- | --- | --- |
| Original EyePACS Kaggle training release | Five-grade model training and separately held-out internal evaluation | [Kaggle competition data](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data) |
| Original Messidor | Required external test under its own R0–R3 grades; use the declared binary referable-DR proxy for cross-source comparison | [ADCIS Messidor](https://www.adcis.net/en/third-party/messidor/) |
| Messidor-2 | Optional second external cohort if official images, pairing metadata, and independent adjudicated grades reconcile | [ADCIS Messidor-2](https://www.adcis.net/en/third-party/messidor2/) and [Google Brain grades](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) |

The [Kaggle preparation and diagnostic notebooks](../README.md#verified-kaggle-work)
are linked with their exact purposes in the README and dossier. The current
source revision, the completed training run, sealed test metrics, prediction
examples, and the final result report have distinct verification gates. Access
to a dataset does not satisfy a model result deliverable.
