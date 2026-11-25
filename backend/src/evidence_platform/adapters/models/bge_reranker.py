import asyncio
import logging
import threading
import torch

try:
    from FlagEmbedding import FlagReranker
    _HAS_RERANKER = True
except ImportError:
    _HAS_RERANKER = False

from ...ports.models import RerankerClient

# Patch Transformers XLMRobertaModel.__init__ to ignore 'dtype' kwarg passed by FlagEmbedding in transformers >= 4.49
try:
    from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel
    _orig_xlm_init = XLMRobertaModel.__init__
    def _patched_xlm_init(self, *args, **kwargs):
        kwargs.pop("dtype", None)
        return _orig_xlm_init(self, *args, **kwargs)
    XLMRobertaModel.__init__ = _patched_xlm_init
except Exception:
    pass

logger = logging.getLogger("evidence_platform.adapters.bge_reranker")

class BGEM3RerankerClient(RerankerClient):
    """Real BGE-Reranker client using local GPU (CUDA) with memory optimization."""

    _reranker = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        if not _HAS_RERANKER:
            raise ImportError("FlagEmbedding library is not installed.")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if BGEM3RerankerClient._reranker is None:
            logger.info(f"Initializing BGE-Reranker model on device: {self.device}")
            BGEM3RerankerClient._reranker = FlagReranker(
                "BAAI/bge-reranker-v2-m3",
                use_fp16=(self.device == "cuda"),
                device=self.device
            )
            logger.info("BGE-Reranker model loaded successfully.")
        self.reranker = BGEM3RerankerClient._reranker

    def _score_sync(self, query: str, passages: list[str]) -> list[float]:
        sentence_pairs = [[query, passage] for passage in passages]
        batch_size = 32
        scores: list[float] = []
        with BGEM3RerankerClient._lock, torch.no_grad():
            for i in range(0, len(sentence_pairs), batch_size):
                batch = sentence_pairs[i : i + batch_size]
                batch_scores = self.reranker.compute_score(batch, normalize=True, max_length=512)
                if isinstance(batch_scores, list):
                    scores.extend(float(s) for s in batch_scores)
                else:
                    scores.append(float(batch_scores))
        return scores

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Compute cross-encoder relevance scores (0-1) for passages against a query.

        Runs in a worker thread so GPU inference does not block the event loop.
        """
        if not passages:
            return []
        logger.info(f"Reranking {len(passages)} passages against query: {query[:60]}...")
        return await asyncio.to_thread(self._score_sync, query, passages)
