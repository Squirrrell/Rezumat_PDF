"""Shared section names and retrieval queries for summarization features."""

from __future__ import annotations

FIELD_RETRIEVAL_QUERIES: dict[str, str] = {
    "Title / Topic": "title abstract topic paper subject",
    "Main idea": "main contribution main idea abstract overview introduction",
    "Problem addressed": "problem motivation challenge gap issue limitation existing methods",
    "Proposed method": "method approach proposed framework architecture algorithm system model",
    "Dataset / experiments": "dataset experiment evaluation benchmark setup data training testing",
    "Results": "results findings performance comparison accuracy improvement evaluation",
    "Limitations": "limitations weakness constraints threats future work discussion",
    "Conclusion": "conclusion final remarks summary implications",
    "Key takeaways": "main findings contributions results conclusion important points",
}

COMPREHENSIVE_SECTIONS: list[str] = list(FIELD_RETRIEVAL_QUERIES.keys())
