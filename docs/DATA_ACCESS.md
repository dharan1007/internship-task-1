# Official data access and staging

Use your own account, identity, organization and acceptance of source terms.
Access links and downloaded archives remain private.

## EyePACS

1. Open the [Kaggle Diabetic Retinopathy Detection data page](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data).
2. Sign in or create a Kaggle account. Accept the competition rules yourself
   if required.
3. Obtain
   `trainLabels.csv.zip` and **all** `train.zip.*` image parts. The
   unlabeled competition `test.zip.*` cannot yield ground-truth metrics.
   Extract the training images into secure cloud storage.
4. Keep archive hashes, source URL and access date in private provenance.
   Point `prepare-eyepacs` at extracted training images and `trainLabels.csv`.

The [official Kaggle CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md)
supports downloading a specific competition file to a chosen directory. It is
an alternative transport for the **same official files**, not a replacement
dataset. The account must authenticate and have accepted the competition rules.
For least privilege, use a token scoped to competition data download if Kaggle
offers that scope; its default CLI OAuth login requests broad account access.
Keep tokens out of this repository and chat.
The original competition rules limit use to the competition's purpose and
duration unless separate permission applies. Joining the closed competition
and obtaining a CLI token do not establish internship reuse rights. An
EyePACS provider reply received on 2026-11-04 directs download
from the public Kaggle page and requests attribution in reports. All five
original training image parts have been downloaded. The reply does not expressly address the separate
training, weight-sharing and report-image permissions requested in the original
message; record any further written clarification before sharing those outputs.

## Original Messidor

1. Open the [official ADCIS Messidor page](https://www.adcis.net/en/third-party/messidor/)
   and scroll to **Request access to the Messidor database**.
2. Enter your own name, email, organization, country and intended
   research/educational use. Read and accept the required licence/privacy
   choices yourself; newsletter consent is optional. Submit the form.
3. ADCIS sends a personal download link valid for seven days. Download all
   12 zipped subsets (three sets of four) with their Excel annotation
   workbooks. Retain the errata page.
4. Extract everything to one private directory and run `prepare-messidor`.
   The parser fails if known correction or duplicate targets are missing.

## Messidor-2 and its labels

1. Open the [official ADCIS Messidor-2 page](https://www.adcis.net/en/third-party/messidor2/)
   and complete its separate form with your own details and acceptance.
2. Download the image archive and official pairing spreadsheet from the
   personal link. ADCIS does **not** include five-grade labels in this release.
   If the form does not produce a link, the [official ADCIS page](https://www.adcis.net/en/third-party/messidor2/)
   instructs users to contact `webmaster@adcis.net` about download trouble.
3. Open the [Google Brain Messidor-2 DR grades page](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades),
   sign in, observe its terms and download `messidor_data.csv` containing
   `image_id`, `adjudicated_dr_grade`, `adjudicated_gradable`.
4. Normalize the official pairing sheet to a private `image_id,subject_id`
   CSV with one row per image and pass it as `--pairs`. The preparation command
   reconciles grades to images and records ungradable exclusions.

Do not send passwords, API keys or personal ADCIS links. Once files are
available, give their private storage paths or mount points for the training
environment; the pipeline reads them without publishing them.
