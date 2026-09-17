# GitHub Milestone Push Workflow

1. Conduct research only in the authoritative `C:/Users/rohan/SHIFT-ICD` repository.
2. Complete the scientific milestone, quality gates, and authoritative commit/tag.
3. Inventory reachable large blobs before publication.
4. Preserve hashes and compact manifests for excluded generated artifacts.
5. Rebuild or filter the separate `C:/Users/rohan/SHIFT-ICD-GITHUB-MIRROR` workspace.
6. Verify that the mirror has no reachable ordinary Git blob at or above 95 MiB.
7. Commit mirror-specific provenance and documentation with the authoritative source SHA.
8. Fetch the intended GitHub remote and stop if unexpected refs exist.
9. Push mirror `main` and its milestone tag without force-push.
10. Verify remote refs and content, then return to the authoritative repository for final health checks.

The authoritative history is never rewritten. Major milestones are published; smoke/debug commits remain local unless separately authorized.
