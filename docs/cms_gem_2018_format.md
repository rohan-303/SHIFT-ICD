# CMS FY 2018 GEM record format

## Official references

- CMS landing page: https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-10-cm-icd-10-pcs-gem-archive
- CMS GEM archive: 2018 General Equivalence Mappings (GEMS) (ZIP)
- CMS technical documentation: `GemsTechDoc_2018.pdf` inside the GEM archive
- CMS diagnosis guide: `Dxgem_guide_2018.pdf` inside the GEM archive
- CMS ICD-10-CM descriptions: https://www.cms.gov/medicare/coding/icd10/downloads/2018-icd-10-code-descriptions.zip
- CMS ICD-9-CM titles: https://www.cms.gov/medicare/coding-billing/icd-10-codes/icd-9-cm-diagnosis-procedure-codes-abbreviated-and-full-code-titles

The CMS archive lists GEM resources for FY 2018 but does not list a 2019 or 2020 ICD-10-CM GEM resource. It separately lists 2019 and 2020 code-description/conversion resources. Within this CMS archive, FY 2018 is therefore the final listed ICD-10-CM diagnosis GEM release. This is a statement about the CMS GEM archive and should not be generalized to every later crosswalk or terminology resource.

## Downloaded archives

| Local archive | Official resource | SHA-256 |
|---|---|---|
| `2018-icd-10-cm-general-equivalence-mappings.zip` | 2018 General Equivalence Mappings (GEMs) | `1c5e5f14026ace48437a0d1c485d282fb5d171c357c6ab4f7798fc0a1e3624a2` |
| `2018-icd-10-code-descriptions.zip` | 2018 Code Descriptions in Tabular Order | `d4954a3fa02e0bfbecb10b20864198dfa50862ded93f2d9d7313e9fe07ef0dfc` |
| `icd-9-cm-v32-master-descriptions.zip` | Version 32 Full and Abbreviated Code Titles | `45a7d05ddcadf124af88375b64cdf068bb1e3f999ce7fdacb91f65f4e6d55f08` |

Archives are preserved without editing under `data/raw/cms/2018_gem/`. The complete provenance record is `data/raw/cms/2018_gem/manifest.json`.

## Archive contents

### GEM archive

| Member | Size | Role |
|---|---:|---|
| `2018_I9gem.txt` | 522,060 bytes | ICD-9-CM source to ICD-10-CM target diagnosis mappings |
| `2018_I10gem.txt` | 1,713,453 bytes | ICD-10-CM source to ICD-9-CM target diagnosis mappings |
| `Dxgem_guide_2018.pdf` | 394,379 bytes | Diagnosis GEM guide |
| `GemsTechDoc_2018.pdf` | 412,137 bytes | Technical FAQ and flag documentation |

### ICD-10-CM descriptions archive

| Member | Size | Role |
|---|---:|---|
| `icd10cm_order_2018.txt` | 14,076,464 bytes | Tabular-order code descriptions |
| `icd10cm_order_addenda_2018.txt` | 189,662 bytes | Tabular-order addenda |
| `icd10cm_codes_2018.txt` | 6,154,304 bytes | Code-description file |
| `icd10cm_codes_addenda_2018.txt` | 102,280 bytes | Code addenda |
| `icd10cmCodesFile.pdf` | 341,631 bytes | PDF representation |
| `icd10OrderFiles.pdf` | 431,216 bytes | PDF representation |

### ICD-9-CM title archive

The archive contains `CMS32_DESC_LONG_DX.txt`, `CMS32_DESC_SHORT_DX.txt`, and `CMS32_DESC_LONG_SHORT_DX.xlsx` for diagnosis titles, plus analogous surgery-title members. The diagnosis title members are relevant to Track A; all members remain preserved in the raw archive.

## Related description-file structures

The `icd10cm_order_2018.txt` member is a CRLF-terminated, fixed-width tabular-order file with embedded hierarchy/instruction rows and human-readable descriptions; it is not a simple two-column CSV. The `icd10cm_codes_2018.txt` member is also fixed-width text and begins with code/title records. The `CMS32_DESC_LONG_DX.txt` ICD-9-CM diagnosis title member is LF-terminated fixed-width text with a six-character, space-padded code field followed by the long description. These files are retained as raw label resources; no label extraction or normalization is performed in this milestone.


The layout is direction-dependent:

| File | Source field | Target field | Flags |
|---|---:|---:|---:|
| `2018_I9gem.txt` | columns 1–6 | columns 7–14 | columns 15–19 |
| `2018_I10gem.txt` | columns 1–8 | columns 9–14 | columns 15–19 |

The source and target fields are space-padded. The source field is six characters for ICD-9-CM and eight characters for ICD-10-CM; the target field uses the complementary width. Codes in the stored GEM rows omit punctuation such as the decimal point. This description records the raw representation only; normalization is deferred.

Examples:

```text
0010  A000    00000
A000    0010  00000
0020  A0100   10000
```

The final five characters are five positional flags, not a single categorical label:

```text
approximate | no-map | combination | scenario | choice-list
```

## Flag definitions

### Flag 1 — approximate

- `0`: the source and target complete meanings are considered equivalent under the GEM methodology.
- `1`: the complete meanings are not considered equivalent. CMS describes this as including cases where one side is more or less specific, or where specificity differs across classification axes.

The flag is about the complete meaning, including relevant instructional notes and index references—not merely whether the visible titles look similar. Identical or nearly identical titles can still have `approximate=1`.

### Flag 2 — no-map

- `0`: at least one target translation exists.
- `1`: no acceptable target translation exists. CMS uses `NoDx` in the diagnosis target field rather than leaving it blank.

A no-map row is therefore an explicit record, not a missing-data row.

### Flag 3 — combination

- `0`: the target is a single complete translation alternative.
- `1`: the target participates in a target-code cluster. Two or more target codes taken together provide the translation.

A source can have both non-combination and combination alternatives. These must not be flattened into a list of unrelated target labels.

### Flag 4 — scenario

A scenario identifies a variation of a combination entry. If a source combination has one variation, its rows use scenario `1`. If it has multiple variations, each alternative cluster is assigned scenario numbers beginning at `1`.

Rows with the same source and scenario belong to the same scenario-level translation structure. Scenario `0` is observed on ordinary non-combination rows and no-map rows.

### Flag 5 — choice list

Choice-list values organize the distinct components of a combination translation into pick lists. One target code from each choice list forms a complete target cluster. A choice list can contain one or multiple choices. It is not equivalent to a simple row number or an independent alternative label.

For example, an entry with choice lists `1`, `2`, and `3` requires selecting one code from each list to reconstruct a complete combination translation. With multiple choices in one list, the complete translations are combinations across the lists.

## Mapping structures

### One-to-one mapping

A source code with one target row and flags `00000` or `10000` is represented by one raw row. For example:

```text
0010  A000    00000
```

This means source `0010` has target `A000`, with no approximate, no-map, or combination flag. It does not imply that the reverse GEM is a mathematical inverse.

### One-to-many mapping

A source code can occur on multiple rows. This may represent multiple translation alternatives, multiple scenario rows, or the components of one or more combination clusters. Therefore “source appears on multiple rows” is a raw structural observation, not by itself the definition of ambiguous clinical meaning.

The initial raw profile observes a maximum of 533 rows associated with one ICD-9-CM source and 12 rows associated with one ICD-10-CM source in the two diagnosis files.

### Approximate mapping

Approximate mappings remain explicit through flag 1. They must not be relabeled as exact matches merely because source and target titles overlap.

### No-map record

Representative raw records include:

```text
7796  NoDx    11000
Z6740   NoDx  11000
```

The first is an ICD-9-CM-to-ICD-10-CM no-map form; the second is an ICD-10-CM-to-ICD-9-CM no-map form. The `NoDx` target token and `no-map=1` jointly convey the explicit absence of an acceptable diagnosis translation.

### Combination mapping

A combination mapping may require a target cluster. CMS’s technical example for ICD-10-CM `I25.111` shows two rows with `10111` and `10112`: both have combination flag `1`, scenario `1`, and choice-list positions `1` and `2`. Together they form one target cluster.

Another CMS example, ICD-9-CM `800.22`, has one target row in choice list `1` and several alternatives in choice list `2`. Selecting one target from each list yields a complete cluster. Treating every row as an independent positive classification target would create invalid partial labels.

### Scenario and choice-list semantics

A raw row is the smallest stored relationship. A source concept is identified by its source code. A target concept is identified by a target code. A mapping group is the higher-level structure needed to reconstruct a complete translation alternative, using source code, scenario, combination flag, and choice-list position.

The future canonical representation must preserve at least:

```text
source code
mapping direction
scenario identifier
choice-list identifier
ordered target rows
approximate flag
no-map flag
combination flag
```

The canonical representation is not created in this milestone.

## Profiling results

The non-transformative profile is stored at `artifacts/data_audit/cms_2018_gem_profile.json` and was generated by `scripts/profile_cms_gem.py`.

| Direction | Raw rows | Unique sources | Unique targets | Sources with multiple rows | Max rows/source |
|---|---:|---:|---:|---:|---:|
| ICD-9-CM → ICD-10-CM | 24,860 | 14,567 | 17,514 | 3,380 | 533 |
| ICD-10-CM → ICD-9-CM | 81,593 | 71,704 | 11,691 | 7,862 | 12 |

Approximate rows are 21,338 forward and 78,071 reverse. No-map rows are 422 forward and 731 reverse. Combination rows are 2,338 forward and 9,033 reverse. Scenario and choice-list distributions are recorded in the machine-readable profile rather than copied incompletely here.

These are raw row statistics, not benchmark sample counts.

## Implications for future benchmark construction

1. The forward and backward files are separate directional resources and must remain separate.
2. A source code may have multiple rows without each row being an independent alternative.
3. Combination mappings require group reconstruction using scenario and choice-list information.
4. No-map and approximate status must remain explicit labels or attributes.
5. Exact-match evaluation must define whether correctness is row-level, target-set-level, cluster-level, or direction-specific.
6. Any future split must prevent scenario fragments from crossing train and test boundaries.
7. Raw code punctuation and field padding must remain untouched until a documented canonicalization step is designed.
8. CMS GEMs are translation reference dictionaries, not guaranteed automated one-to-one conversion rules.
