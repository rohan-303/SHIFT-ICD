# Data sources and governance

## Source priority

Authoritative terminology data takes priority. Initial source organizations are:

- Centers for Medicare & Medicaid Services (CMS);
- Centers for Disease Control and Prevention / National Center for Health Statistics (CDC/NCHS), where applicable;
- World Health Organization (WHO).

Unofficial Kaggle mirrors, scraped copies, or GitHub copies must not silently replace authoritative sources. If a mirror is used temporarily for convenience, its contents must later be verified against the authoritative resource and the discrepancy record retained.

## Required provenance record

For every future dataset or terminology resource, record:

- source organization;
- exact dataset/resource name;
- URL;
- ICD version and release;
- download date;
- license and terms of use;
- checksum when downloaded;
- raw filename;
- preprocessing script;
- transformations performed;
- source format and schema notes;
- verification status against the authoritative source.

## Data handling rules

Raw data directories are immutable after acquisition. Never edit downloaded files in place. Store transformations in versioned scripts and write outputs to interim or processed locations. Do not commit restricted data, large datasets, credentials, or generated model artifacts.

Each transformation must document whether it changes labels, mapping cardinality, hierarchy, missingness, release identity, or the set of valid candidate targets. Missing information must remain explicitly missing; unsupported mappings must not be invented.

## Obtained CMS resources for Step 2

The following authoritative CMS resources were obtained on 2026-08-28 and preserved as immutable archives:

| Source organization | Resource title | Official URL | Version/release | Local raw path | SHA-256 | Acquisition script |
|---|---|---|---|---|---|---|
| CMS | 2018 General Equivalence Mappings (GEMs) | https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-cm-general-equivalence-mappings.zip | FY 2018 | `data/raw/cms/2018_gem/2018-icd-10-cm-general-equivalence-mappings.zip` | `1c5e5f14026ace48437a0d1c485d282fb5d171c357c6ab4f7798fc0a1e3624a2` | `scripts/download_cms_gem.py` |
| CMS | 2018 Code Descriptions in Tabular Order | https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-code-descriptions.zip | FY 2018 ICD-10-CM | `data/raw/cms/2018_gem/2018-icd-10-code-descriptions.zip` | `d4954a3fa02e0bfbecb10b20864198dfa50862ded93f2d9d7313e9fe07ef0dfc` | `scripts/download_cms_gem.py` |
| CMS | Version 32 Full and Abbreviated Code Titles | https://www.cms.gov/medicare/coding/icd9providerdiagnosticcodes/downloads/icd-9-cm-v32-master-descriptions.zip | Version 32; effective October 1, 2014 | `data/raw/cms/2018_gem/icd-9-cm-v32-master-descriptions.zip` | `45a7d05ddcadf124af88375b64cdf068bb1e3f999ce7fdacb91f65f4e6d55f08` | `scripts/download_cms_gem.py` |

Official landing pages used were the [CMS ICD-10 Files & News Archive](https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-10-cm-icd-10-pcs-gem-archive) and the [CMS ICD-9-CM Diagnosis and Procedure Codes page](https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles). The complete machine-readable manifest, including archive members, sizes, terms, and notes, is `data/raw/cms/2018_gem/manifest.json`.

CMS’s inspected archive includes both `2018_I9gem.txt` (ICD-9-CM → ICD-10-CM) and `2018_I10gem.txt` (ICD-10-CM → ICD-9-CM), plus `Dxgem_guide_2018.pdf` and `GemsTechDoc_2018.pdf`. The diagnosis GEM text records are fixed-width, 19-character, ASCII-compatible lines without headers. Full format documentation is in `docs/cms_gem_2018_format.md`.

The archive lists the FY 2018 GEM package but no 2019 or 2020 ICD-10-CM GEM package; it lists later code-description and conversion resources separately. FY 2018 is therefore the final ICD-10-CM diagnosis GEM release listed in the inspected CMS archive, not a claim that no later crosswalk resource exists anywhere.

The inspected CMS landing page does not state a separate open-source license for these downloads. The files are publicly distributed by CMS, but redistribution and derivative-use terms must be checked before any external release. Raw files are therefore excluded from Git and retained locally with checksums.

## Planned later sources

Track A will continue with the obtained CMS ICD-9-CM → ICD-10-CM GEMs and official code-description resources after the canonical representation is designed.

Track B and Track C will use WHO resources only after their release structure, API/download terms, available mapping tables, Foundation/MMS distinctions, and postcoordination representation are verified.

## Human and clinical data boundary

The initial project uses public terminology resources and does not require private clinical notes or protected health information. Any future use of protected/private clinical data requires a separate approved protocol, data-use authorization, privacy review, and explicit documentation before acquisition or processing.
