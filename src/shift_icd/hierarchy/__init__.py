"""Leakage-safe ontology metadata and structural reranking primitives."""

from .metadata import HierarchyRecord, audit_hierarchy, build_prefix_hierarchy

__all__ = ["HierarchyRecord", "audit_hierarchy", "build_prefix_hierarchy"]
